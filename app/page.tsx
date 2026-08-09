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

type FaceLandmarkerInstance = {
  detectForVideo: (
    video: HTMLVideoElement,
    timestampMs: number,
  ) => { faceLandmarks?: Landmark[][] };
  close: () => void;
};

const VISION_WASM = "/mediapipe/wasm";
const HAND_MODEL = "/mediapipe/hand_landmarker.task";
const FACE_MODEL = "/mediapipe/face_landmarker.task";
const EFFECT_FALLBACK_DURATION_MS = 10_500;

function distance(a: Landmark, b: Landmark) {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

function distance2d(a: Landmark, b: Landmark) {
  return Math.hypot(a.x - b.x, a.y - b.y);
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

function averageLandmarks(points: Landmark[]) {
  const total = points.reduce(
    (sum, point) => ({
      x: sum.x + point.x,
      y: sum.y + point.y,
      z: sum.z + point.z,
    }),
    { x: 0, y: 0, z: 0 },
  );

  return {
    x: total.x / points.length,
    y: total.y / points.length,
    z: total.z / points.length,
  };
}

function isNoseCovered(hand: Landmark[], face: Landmark[]) {
  if (hand.length < 21 || face.length < 455) return false;

  const nose = face[1];
  const faceWidth = distance2d(face[234], face[454]);
  if (faceWidth < 0.08) return false;

  const palmAnchors = [hand[0], hand[5], hand[9], hand[13], hand[17]];
  const palmCenter = averageLandmarks(palmAnchors);
  const nearbyPoints = hand.filter(
    (point) => distance2d(point, nose) < faceWidth * 0.24,
  ).length;
  const palmIsCentered = distance2d(palmCenter, nose) < faceWidth * 0.3;

  const pinchCenter = averageLandmarks([hand[4], hand[8]]);
  const pinchIsAtNose =
    distance2d(pinchCenter, nose) < faceWidth * 0.2 &&
    distance2d(hand[4], hand[8]) < faceWidth * 0.32;

  return (palmIsCentered && nearbyPoints >= 3) || (pinchIsAtNose && nearbyPoints >= 2);
}

function captureFrame(video: HTMLVideoElement) {
  if (!video.videoWidth || !video.videoHeight) return null;

  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const context = canvas.getContext("2d");
  if (!context) return null;

  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.9);
}

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const handLandmarkerRef = useRef<HandLandmarkerInstance | null>(null);
  const faceLandmarkerRef = useRef<FaceLandmarkerInstance | null>(null);
  const animationRef = useRef<number | null>(null);
  const effectTimerRef = useRef<number | null>(null);
  const lastVideoTimeRef = useRef(-1);
  const lastDetectionAtRef = useRef(0);
  const matchFramesRef = useRef(0);
  const missFramesRef = useRef(0);
  const noseMatchFramesRef = useRef(0);
  const isGestureRef = useRef(false);
  const isNoseEffectActiveRef = useRef(false);
  const didAutoStartRef = useRef(false);

  const [cameraState, setCameraState] = useState<CameraState>("idle");
  const [modelState, setModelState] = useState<ModelState>("idle");
  const [isGesture, setIsGesture] = useState(false);
  const [isNoseEffectActive, setIsNoseEffectActive] = useState(false);
  const [frozenFrame, setFrozenFrame] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState("");

  const loadVisionModels = useCallback(async () => {
    if (
      (handLandmarkerRef.current && faceLandmarkerRef.current) ||
      modelState === "loading"
    ) {
      return;
    }

    setModelState("loading");
    try {
      const vision = await import("@mediapipe/tasks-vision");
      const fileset = await vision.FilesetResolver.forVisionTasks(VISION_WASM);

      if (!handLandmarkerRef.current) {
        const handOptions = {
          runningMode: "VIDEO",
          numHands: 1,
          minHandDetectionConfidence: 0.55,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        } as const;

        try {
          handLandmarkerRef.current =
            await vision.HandLandmarker.createFromOptions(fileset, {
              ...handOptions,
              baseOptions: { modelAssetPath: HAND_MODEL, delegate: "GPU" },
            });
        } catch (gpuError) {
          console.warn(
            "Akselerasi GPU untuk tangan tidak tersedia, beralih ke CPU",
            gpuError,
          );
          handLandmarkerRef.current =
            await vision.HandLandmarker.createFromOptions(fileset, {
              ...handOptions,
              baseOptions: { modelAssetPath: HAND_MODEL },
            });
        }
      }

      if (!faceLandmarkerRef.current) {
        const faceOptions = {
          runningMode: "VIDEO",
          numFaces: 1,
          minFaceDetectionConfidence: 0.55,
          minFacePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
          outputFaceBlendshapes: false,
          outputFacialTransformationMatrixes: false,
        } as const;

        try {
          faceLandmarkerRef.current =
            await vision.FaceLandmarker.createFromOptions(fileset, {
              ...faceOptions,
              baseOptions: { modelAssetPath: FACE_MODEL, delegate: "GPU" },
            });
        } catch (gpuError) {
          console.warn(
            "Akselerasi GPU untuk wajah tidak tersedia, beralih ke CPU",
            gpuError,
          );
          faceLandmarkerRef.current =
            await vision.FaceLandmarker.createFromOptions(fileset, {
              ...faceOptions,
              baseOptions: { modelAssetPath: FACE_MODEL },
            });
        }
      }

      setModelState("ready");
    } catch (error) {
      console.error("Gagal memuat detektor gesture", error);
      setModelState("error");
    }
  }, [modelState]);

  const startCamera = useCallback(async () => {
    if (isNoseEffectActiveRef.current) return;

    setCameraState("starting");
    setErrorMessage("");
    setFrozenFrame(null);
    setIsGesture(false);
    isGestureRef.current = false;
    matchFramesRef.current = 0;
    missFramesRef.current = 0;
    noseMatchFramesRef.current = 0;
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
      void loadVisionModels();
    } catch (error) {
      console.error("Gagal membuka kamera", error);
      setCameraState("error");
      setErrorMessage(
        "Kamera tidak dapat dibuka. Izinkan akses kamera, lalu coba lagi.",
      );
    }
  }, [loadVisionModels]);

  const finishNoseEffect = useCallback(() => {
    if (!isNoseEffectActiveRef.current) return;

    isNoseEffectActiveRef.current = false;
    if (effectTimerRef.current !== null) {
      window.clearTimeout(effectTimerRef.current);
      effectTimerRef.current = null;
    }

    const audio = audioRef.current;
    if (audio) {
      audio.pause();
      audio.currentTime = 0;
    }

    noseMatchFramesRef.current = 0;
    setIsNoseEffectActive(false);
    setFrozenFrame(null);
    void startCamera();
  }, [startCamera]);

  const triggerNoseEffect = useCallback(
    (video: HTMLVideoElement) => {
      if (isNoseEffectActiveRef.current) return;

      isNoseEffectActiveRef.current = true;
      setFrozenFrame(captureFrame(video));
      setIsNoseEffectActive(true);
      setIsGesture(false);
      isGestureRef.current = false;

      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      video.srcObject = null;
      setCameraState("idle");

      const audio = audioRef.current;
      let effectDuration = EFFECT_FALLBACK_DURATION_MS;
      if (audio) {
        audio.currentTime = 0;
        if (Number.isFinite(audio.duration) && audio.duration > 0) {
          effectDuration = audio.duration * 1000 + 600;
        }
        void audio.play().catch((error) => {
          console.warn("Audio diblokir browser sampai layar disentuh", error);
        });
      }

      effectTimerRef.current = window.setTimeout(
        finishNoseEffect,
        effectDuration,
      );
    },
    [finishNoseEffect],
  );

  useEffect(() => {
    if (didAutoStartRef.current) return;
    didAutoStartRef.current = true;
    void startCamera();
  }, [startCamera]);

  useEffect(() => {
    const unlockAudio = () => {
      const audio = audioRef.current;
      if (!audio || isNoseEffectActiveRef.current) return;

      audio.muted = true;
      void audio
        .play()
        .then(() => {
          audio.pause();
          audio.currentTime = 0;
          audio.muted = false;
        })
        .catch(() => {
          audio.muted = false;
        });
    };

    window.addEventListener("pointerdown", unlockAudio, { once: true });
    return () => window.removeEventListener("pointerdown", unlockAudio);
  }, []);

  useEffect(() => {
    if (
      cameraState !== "active" ||
      modelState !== "ready" ||
      isNoseEffectActive
    ) {
      return;
    }

    const detect = () => {
      const video = videoRef.current;
      const handLandmarker = handLandmarkerRef.current;
      const faceLandmarker = faceLandmarkerRef.current;

      if (
        video &&
        handLandmarker &&
        faceLandmarker &&
        video.readyState >= 2
      ) {
        const now = performance.now();
        if (
          lastVideoTimeRef.current !== video.currentTime &&
          now - lastDetectionAtRef.current >= 160
        ) {
          lastVideoTimeRef.current = video.currentTime;
          lastDetectionAtRef.current = now;
          const handResult = handLandmarker.detectForVideo(video, now);
          const faceResult = faceLandmarker.detectForVideo(video, now);
          const hand = handResult.landmarks?.[0];
          const face = faceResult.faceLandmarks?.[0];

          if (hand && face && isNoseCovered(hand, face)) {
            noseMatchFramesRef.current += 1;
            if (noseMatchFramesRef.current >= 3) {
              triggerNoseEffect(video);
              return;
            }
          } else {
            noseMatchFramesRef.current = 0;
          }

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
  }, [cameraState, isNoseEffectActive, modelState, triggerNoseEffect]);

  useEffect(() => {
    const audio = audioRef.current;

    return () => {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      handLandmarkerRef.current?.close();
      faceLandmarkerRef.current?.close();
      audio?.pause();
      if (animationRef.current !== null) {
        cancelAnimationFrame(animationRef.current);
      }
      if (effectTimerRef.current !== null) {
        window.clearTimeout(effectTimerRef.current);
      }
    };
  }, []);

  const stopCamera = useCallback(() => {
    if (isNoseEffectActiveRef.current) return;

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setIsGesture(false);
    setFrozenFrame(null);
    isGestureRef.current = false;
    matchFramesRef.current = 0;
    missFramesRef.current = 0;
    noseMatchFramesRef.current = 0;
    setCameraState("idle");
    setErrorMessage("");
  }, []);

  const toggleCamera = () => {
    if (cameraState === "active") {
      stopCamera();
      return;
    }

    void startCamera();
  };

  const cameraButtonLabel = isNoseEffectActive
    ? "Kamera dijeda selama efek diputar"
    : cameraState === "active"
      ? "Matikan kamera"
      : cameraState === "starting"
        ? "Kamera sedang dinyalakan"
        : errorMessage
          ? `${errorMessage} Coba hidupkan kamera lagi`
          : "Hidupkan kamera";

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
      {frozenFrame && (
        <div
          className="frozen-frame"
          style={{ backgroundImage: `url(${frozenFrame})` }}
          aria-hidden="true"
        />
      )}
      {isNoseEffectActive && (
        <div
          className="cat-corners"
          role="img"
          aria-label="Empat kucing menari di setiap sudut"
        >
          <span className="cat-corner cat-corner--top-left" />
          <span className="cat-corner cat-corner--top-right" />
          <span className="cat-corner cat-corner--bottom-left" />
          <span className="cat-corner cat-corner--bottom-right" />
        </div>
      )}
      <audio
        ref={audioRef}
        src="/nose-sound.mp3"
        preload="auto"
        onEnded={finishNoseEffect}
      >
        <track
          kind="captions"
          src="/nose-sound.vtt"
          srcLang="id"
          label="Efek suara"
        />
      </audio>
      <button
        className={`camera-toggle ${cameraState === "active" ? "is-active" : ""} ${cameraState === "starting" ? "is-starting" : ""} ${isNoseEffectActive ? "is-effect-paused" : ""}`}
        type="button"
        onClick={toggleCamera}
        disabled={cameraState === "starting" || isNoseEffectActive}
        aria-label={cameraButtonLabel}
        title={cameraButtonLabel}
      >
        <span aria-hidden="true" />
      </button>
      <small className="creator-credit">made by anangfath_</small>
    </main>
  );
}
