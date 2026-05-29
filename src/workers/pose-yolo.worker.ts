/// <reference lib="webworker" />
import * as ort from "onnxruntime-web";
import {
  YOLO_INPUT_SIZE,
  buildTensorFromRgba,
  postprocessYolo,
  type Kp,
} from "../lib/yolo-pose-core";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency ?? 4);
ort.env.wasm.simd = true;

let session: ort.InferenceSession | null = null;
let preprocessBuf: Float32Array | null = null;
let offscreen: OffscreenCanvas | null = null;

type WorkerIn =
  | { type: "init"; modelUrl: string }
  | { type: "infer"; id: number; bitmap: ImageBitmap; vw: number; vh: number; conf: number };

type WorkerOut =
  | { type: "ready" }
  | { type: "error"; message: string }
  | { type: "result"; id: number; kps: Kp[] | null; ms: number };

async function initSession(modelUrl: string) {
  const providers: ort.InferenceSession.SessionOptions["executionProviders"] = ["wasm"];
  try {
    if (typeof navigator !== "undefined" && "gpu" in navigator) {
      providers.unshift("webgpu");
    }
  } catch {
    /* webgpu unavailable */
  }
  session = await ort.InferenceSession.create(modelUrl, { executionProviders: providers });
  offscreen = new OffscreenCanvas(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);
}

self.onmessage = (ev: MessageEvent<WorkerIn>) => {
  const msg = ev.data;

  if (msg.type === "init") {
    void initSession(msg.modelUrl)
      .then(() => self.postMessage({ type: "ready" } satisfies WorkerOut))
      .catch((e: Error) =>
        self.postMessage({ type: "error", message: e.message } satisfies WorkerOut),
      );
    return;
  }

  if (msg.type === "infer") {
    const t0 = performance.now();
    void (async () => {
      try {
        if (!session || !offscreen) throw new Error("Model not loaded");

        const ctx = offscreen.getContext("2d")!;
        const scale = Math.min(YOLO_INPUT_SIZE / msg.vw, YOLO_INPUT_SIZE / msg.vh);
        const nw = Math.round(msg.vw * scale);
        const nh = Math.round(msg.vh * scale);
        const padX = (YOLO_INPUT_SIZE - nw) / 2;
        const padY = (YOLO_INPUT_SIZE - nh) / 2;

        ctx.fillStyle = "#808080";
        ctx.fillRect(0, 0, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);
        ctx.drawImage(msg.bitmap, padX, padY, nw, nh);
        msg.bitmap.close();

        const { data } = ctx.getImageData(0, 0, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);
        const { tensor, buffer } = buildTensorFromRgba(data, preprocessBuf);
        preprocessBuf = buffer;

        const yoloOut = await session.run({
          images: new ort.Tensor("float32", tensor, [1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE]),
        });
        const key = Object.keys(yoloOut)[0];
        const tensorOut = yoloOut[key];
        const kps = postprocessYolo(
          tensorOut.data as Float32Array,
          tensorOut.dims,
          msg.conf,
          scale,
          padX,
          padY,
          msg.vw,
          msg.vh,
        );

        self.postMessage({
          type: "result",
          id: msg.id,
          kps,
          ms: Math.round(performance.now() - t0),
        } satisfies WorkerOut);
      } catch (e) {
        msg.bitmap.close();
        self.postMessage({
          type: "error",
          message: e instanceof Error ? e.message : String(e),
        } satisfies WorkerOut);
      }
    })();
  }
};
