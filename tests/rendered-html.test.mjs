import assert from "node:assert/strict";
import { stat } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("merender halaman kamera VBlur", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>VBlur/);
  assert.match(html, /Tampilan kamera live/);
  assert.match(html, /Hidupkan kamera/);
  assert.match(html, /made by anangfath_/);
  assert.match(html, /nose-sound\.mp3/);
  assert.doesNotMatch(html, /TUNJUKKAN V|Tidak direkam|Balik|Buka kamera/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton/i);
});

test("menyertakan model gesture secara lokal", async () => {
  const assets = [
    "../public/mediapipe/hand_landmarker.task",
    "../public/mediapipe/face_landmarker.task",
    "../public/mediapipe/wasm/vision_wasm_internal.js",
    "../public/mediapipe/wasm/vision_wasm_internal.wasm",
    "../public/mediapipe/wasm/vision_wasm_nosimd_internal.js",
    "../public/mediapipe/wasm/vision_wasm_nosimd_internal.wasm",
  ];

  for (const asset of assets) {
    const file = await stat(new URL(asset, import.meta.url));
    assert.ok(file.size > 100_000, `${asset} tidak lengkap`);
  }
});

test("menyertakan media efek hidung", async () => {
  const assets = ["../public/cat-nose.webp", "../public/nose-sound.mp3"];

  for (const asset of assets) {
    const file = await stat(new URL(asset, import.meta.url));
    assert.ok(file.size > 100_000, `${asset} tidak lengkap`);
  }
});
