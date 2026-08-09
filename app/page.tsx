"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type CameraState = "idle" | "starting" | "active" | "error";
type ModelState = "idle" | "loading" | "ready" | "error";
type Landmark = { x: number; y: number; z: number };

type HandLandmarkerInstance = {
  detectForVideo: (
    video: HTMLVideoElement,
    timestampMs: number,
  ) => { landmarks?: Landmark[][] };
  close: () => void;
};

const VISION_WASM = "/mediapipe/wasm";
const HAND_MODEL = "/mediapipe/hand_landmarker.task";

function distance(a: Landmark, b: Landmark) {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

function jointAngle(a: Landmark, b: Landmark, c: Landmark) {
  const ab = { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
  const cb = { x: c.x - b.x, y: c.y - b.y, z: c.z - b.z };
  const dot = ab.x * cb.x + ab.y * cb.y + ab.z * cb.z;
  const length =
    Math.hypot(ab.x, ab.y, ab.z) * Math.hypot(cb.x, cb.y, cb.z);

  if (!length) return 0;
  return (
    (Math.acos(Math.max(-1, Math.min(1, dot / length))) * 180) / Math.PI
  );
}

function fingerIsExtended(
  points: Landmark[],
  mcp: number,
  pip: number,
  tip: number,
) {
  const wrist = points[0];
  const straight = jointAngle(points[mcp], points[pip], points[tip]) > 150;
  const reachesOut =
    distance(wrist, points[tip]) > distance(wrist, points[pip]) * 1.12;
  return straight && reachesOut;
}

function isVictoryGesture(points: Landmark[]) {
  if (points.length < 21) return false;

  const indexUp = fingerIsExtended(points, 5, 6, 8);
  const middleUp = fingerIsExtended(points, 9, 10, 12);
  const ringDown = !fingerIsExtended(points, 13, 14, 16);
  const pinkyDown = !fingerIsExtended(points, 17, 18, 20);
  const palmWidth = Math.max(distance(points[5], points[17]), 0.01);
  const tipGap = distance(points[8], points[12]);
  const pipGap = distance(points[6], points[10]);
  const fingersFormV = tipGap / palmWidth > 0.34 && tipGap > pipGap * 1.22;

  return indexUp && middleUp && ringDown && pinkyDown && fingersFormV;
}

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<HandLandmarkerInstance | null>(null);
  const animationRef = useRef<number | null>(null);
  const lastVideoTimeRef = useRef(-1);
  const lastDetectionAtRef = useRef(0);
  const matchFramesRef = useRef(0);
  const missFramesRef = useRef(0);
  const isGestureRef = useRef(false);
  const didAutoStartRef = useRef(false);

  const [cameraState, setCameraState] = useState<CameraState>("idle");
  const [modelState, setModelState] = useState<ModelState>("idle");
  const [isGesture, setIsGesture] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const loadGestureModel = useCallback(async () => {
    if (landmarkerRef.current || modelState === "loading") return;

    setModelState("loading");
    try {
      const vision = await import("@mediapipe/tasks-vision");
      const fileset = await vision.FilesetResolver.forVisionTasks(VISION_WASM);
      const options = {
        baseOptions: { modelAssetPath: HAND_MODEL },
        runningMode: "VIDEO",
        numHands: 1,
        minHandDetectionConfidence: 0.55,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      } as const;

      try {
        landmarkerRef.current = await vision.HandLandmarker.createFromOptions(
          fileset,
          {
            ...options,
            baseOptions: { ...options.baseOptions, delegate: "GPU" },
          },
        );
      } catch (gpuError) {
        console.warn("Akselerasi GPU tidak tersedia, beralih ke CPU", gpuError);
        landmarkerRef.current = await vision.HandLandmarker.createFromOptions(
          fileset,
          options,
        );
      }

      setModelState("ready");
    } catch (error) {
      console.error("Gagal memuat detektor gesture", error);
      setModelState("error");
    }
  }, [modelState]);

  const startCamera = useCallback(async () => {
    setCameraState("starting");
    setErrorMessage("");
    setIsGesture(false);
    isGestureRef.current = false;
    matchFramesRef.current = 0;
    missFramesRef.current = 0;
    streamRef.current?.getTracks().forEach((track) => track.stop());

    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraState("error");
      setErrorMessage("Browser ini belum mendukung akses kamera.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: "user" },
          width: { ideal: 1920 },
          height: { ideal: 1080 },
        },
      });

      streamRef.current = stream;
      const video = videoRef.current;
      if (video) {
        video.srcObject = stream;
        await video.play();
      }

      setCameraState("active");
      void loadGestureModel();
    } catch (error) {
      console.error("Gagal membuka kamera", error);
      setCameraState("error");
      setErrorMessage(
        "Kamera tidak dapat dibuka. Izinkan akses kamera, lalu coba lagi.",
      );
    }
  }, [loadGestureModel]);

  useEffect(() => {
    if (didAutoStartRef.current) return;
    didAutoStartRef.current = true;
    void startCamera();
  }, [startCamera]);

  useEffect(() => {
    if (cameraState !== "active" || modelState !== "ready") return;

    const detect = () => {
      const video = videoRef.current;
      const landmarker = landmarkerRef.current;

      if (video && landmarker && video.readyState >= 2) {
        const now = performance.now();
        if (
          lastVideoTimeRef.current !== video.currentTime &&
          now - lastDetectionAtRef.current >= 160
        ) {
          lastVideoTimeRef.current = video.currentTime;
          lastDetectionAtRef.current = now;
          const result = landmarker.detectForVideo(video, now);
          const hand = result.landmarks?.[0];

          if (hand && isVictoryGesture(hand)) {
            matchFramesRef.current += 1;
            missFramesRef.current = 0;
            if (matchFramesRef.current >= 2 && !isGestureRef.current) {
              isGestureRef.current = true;
              setIsGesture(true);
            }
          } else {
            matchFramesRef.current = 0;
            missFramesRef.current += 1;
            if (missFramesRef.current >= 4 && isGestureRef.current) {
              isGestureRef.current = false;
              setIsGesture(false);
            }
          }
        }
      }

      animationRef.current = requestAnimationFrame(detect);
    };

    animationRef.current = requestAnimationFrame(detect);
    return () => {
      if (animationRef.current !== null) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [cameraState, modelState]);

  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      landmarkerRef.current?.close();
      if (animationRef.current !== null) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  return (
    <main className={`camera-app ${isGesture ? "is-blurred" : ""}`}>
      <video
        ref={videoRef}
        className="camera-feed is-mirrored"
        autoPlay
        muted
        playsInline
        aria-label="Tampilan kamera live"
      />
      {cameraState === "error" && (
        <button
          className="camera-retry"
          type="button"
          onClick={() => void startCamera()}
        >
          <span>{errorMessage}</span>
          Coba lagi
        </button>
      )}
    </main>
  );
}
