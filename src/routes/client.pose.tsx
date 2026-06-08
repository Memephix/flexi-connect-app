import { createFileRoute } from "@tanstack/react-router";
import { RoleGuard } from "@/components/auth/RoleGuard";
import { useAuth } from "@/hooks/use-auth";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";
import { useState, useRef, useEffect, useCallback } from "react";
import {
  Play, Square, CheckCircle2, Upload, FileVideo,
  Cpu, Target, RotateCcw, Loader2, AlertTriangle,
} from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import {
  type ExerciseMode,
  EXERCISE_OPTIONS,
  exerciseUsesOnnx,
  exerciseLabel,
  createSquatRepCounter,
  ExerciseEngine,
} from "@/lib/exercise-eval";
import * as ort from "onnxruntime-web";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency ?? 4);
ort.env.wasm.simd = true;

export const Route = createFileRoute("/client/pose")({
  component: () => (
    <RoleGuard role="client">
      <PoseAnalyzer />
    </RoleGuard>
  ),
});

// ─── Constants ────────────────────────────────────────────────────────────────

const MODEL_BASE = "/models";
const CONF_THRESH = 0.2;
const DISPLAY_LERP = 0.75;
const PREDICT_MS = 80; // extrapolate skeleton ahead to compensate inference delay
const CLASSIFY_EVERY_N = 4;
const UI_UPDATE_MS = 200;

/** COCO 17-keypoint indices */
const KP = {
  L_SHOULDER: 5, R_SHOULDER: 6,
  L_ELBOW: 7,    R_ELBOW: 8,
  L_WRIST: 9,    R_WRIST: 10,
  L_HIP: 11,     R_HIP: 12,
  L_KNEE: 13,    R_KNEE: 14,
  L_ANKLE: 15,   R_ANKLE: 16,
} as const;

const SKELETON: [number, number][] = [
  [KP.L_SHOULDER, KP.R_SHOULDER],
  [KP.L_SHOULDER, KP.L_ELBOW], [KP.L_ELBOW, KP.L_WRIST],
  [KP.R_SHOULDER, KP.R_ELBOW], [KP.R_ELBOW, KP.R_WRIST],
  [KP.L_SHOULDER, KP.L_HIP],  [KP.R_SHOULDER, KP.R_HIP],
  [KP.L_HIP, KP.R_HIP],
  [KP.L_HIP, KP.L_KNEE],      [KP.L_KNEE, KP.L_ANKLE],
  [KP.R_HIP, KP.R_KNEE],      [KP.R_KNEE, KP.R_ANKLE],
];

// ─── Types ────────────────────────────────────────────────────────────────────

interface Kp { x: number; y: number; conf: number }

interface ModelMeta {
  pose_features: string[];
  form_features: string[];
  pose_classes: string[];
  form_classes: string[];
}

interface FrameResult {
  exercise: string;
  poseConf: number;
  form: string;
  formConf: number;
  hasError: boolean;
  features: Record<string, number>;
}

// ─── Geometry ────────────────────────────────────────────────────────────────

function calcAngle(
  a: [number, number],
  b: [number, number],
  c: [number, number],
): number {
  const ba = [a[0] - b[0], a[1] - b[1]];
  const bc = [c[0] - b[0], c[1] - b[1]];
  const dot = ba[0] * bc[0] + ba[1] * bc[1];
  const magBa = Math.sqrt(ba[0] ** 2 + ba[1] ** 2) + 1e-7;
  const magBc = Math.sqrt(bc[0] ** 2 + bc[1] ** 2) + 1e-7;
  return (Math.acos(Math.max(-1, Math.min(1, dot / (magBa * magBc)))) * 180) / Math.PI;
}

function extractFeatures(kps: Kp[]): Record<string, number> {
  const f: Record<string, number> = {};
  const ok = (...idxs: number[]) => idxs.every((i) => kps[i]?.conf > 0);
  const xy = (i: number): [number, number] => [kps[i].x, kps[i].y];

  if (ok(11, 13, 15)) f["left_knee"]  = calcAngle(xy(11), xy(13), xy(15));
  if (ok(12, 14, 16)) f["right_knee"] = calcAngle(xy(12), xy(14), xy(16));
  if (ok(5,  11, 13)) f["left_hip"]   = calcAngle(xy(5),  xy(11), xy(13));
  if (ok(6,  12, 14)) f["right_hip"]  = calcAngle(xy(6),  xy(12), xy(14));
  if ("left_knee" in f && "right_knee" in f)
    f["knee_symmetry"] = Math.abs(f["left_knee"] - f["right_knee"]);

  if (ok(5,  11, 13)) f["trunk_lean"]        = calcAngle(xy(5),  xy(11), xy(13));
  if (ok(13, 15, 11)) f["left_ankle_flex"]   = calcAngle(xy(13), xy(15), xy(11));
  if (ok(14, 16, 12)) f["right_ankle_flex"]  = calcAngle(xy(14), xy(16), xy(12));
  if (ok(13, 15))     f["left_knee_forward"] = kps[13].x - kps[15].x;
  if (ok(14, 16))     f["right_knee_forward"]= kps[14].x - kps[16].x;
  if ("left_hip" in f && "right_hip" in f)
    f["hip_symmetry"] = Math.abs(f["left_hip"] - f["right_hip"]);

  return f;
}

/** Map video-pixel keypoints → canvas overlay coordinates. */
function videoKpsToCanvas(kps: Kp[], vw: number, vh: number, cvW: number, cvH: number): Kp[] {
  if (!vw || !vh || (cvW === vw && cvH === vh)) return kps;
  const sx = cvW / vw;
  const sy = cvH / vh;
  return kps.map((k) => ({ x: k.x * sx, y: k.y * sy, conf: k.conf }));
}

type Vel = { vx: number; vy: number };

function lerpKeypoints(current: Kp[] | null, target: Kp[], factor: number): Kp[] {
  if (!current) return target.map((k) => ({ ...k }));
  return target.map((t, i) => {
    const c = current[i];
    if (!c || t.conf < 0.2) return { ...t };
    return {
      x: c.x + (t.x - c.x) * factor,
      y: c.y + (t.y - c.y) * factor,
      conf: t.conf,
    };
  });
}

function updateVelocity(prev: Kp[] | null, next: Kp[], prevT: number, now: number): Vel[] | null {
  if (!prev || now <= prevT) return null;
  const dt = (now - prevT) / 1000;
  if (dt <= 0) return null;
  return next.map((n, i) => ({
    vx: (n.x - prev[i].x) / dt,
    vy: (n.y - prev[i].y) / dt,
  }));
}

function applyPrediction(kps: Kp[], vel: Vel[] | null, aheadSec: number): Kp[] {
  if (!vel || aheadSec <= 0) return kps;
  return kps.map((k, i) => ({
    x: k.x + vel[i].vx * aheadSec,
    y: k.y + vel[i].vy * aheadSec,
    conf: k.conf,
  }));
}

// ─── Model Runner ─────────────────────────────────────────────────────────────

async function runClassifiers(
  kps: Kp[],
  poseSession: ort.InferenceSession,
  formSession: ort.InferenceSession,
  meta: ModelMeta,
): Promise<FrameResult> {
  const features = extractFeatures(kps);

  const toVector = (cols: string[], fillVal: number) =>
    new Float32Array(cols.map((c) => features[c] ?? fillVal));

  const result: FrameResult = {
    exercise: "Unknown",
    poseConf: 0,
    form: "N/A",
    formConf: 0,
    hasError: false,
    features,
  };

  // ── helper: ดึง key ที่ถูกต้องจาก output object ──────────────────────────
  const getLabelKey = (keys: string[]) =>
    keys.find((k) => k.toLowerCase().includes("label")) ?? keys[0];
  const getProbKey  = (keys: string[]) =>
    keys.find((k) => k.toLowerCase().includes("prob"))  ?? keys[1] ?? keys[0];

  // Model 1: classify pose
  const hasPoseFeats = meta.pose_features.every((c) => c in features);
  if (hasPoseFeats) {
    const poseFeat  = toVector(meta.pose_features, 180);
    const poseInput = { float_input: new ort.Tensor("float32", poseFeat, [1, meta.pose_features.length]) };

    // ✅ FIX: run() returns object — ไม่ใช่ array ห้าม destructure
    const poseOut   = await poseSession.run(poseInput);
    const pKeys     = Object.keys(poseOut);
    result.exercise = String(poseOut[getLabelKey(pKeys)].data[0]);
    const probData  = poseOut[getProbKey(pKeys)].data as Float32Array;
    result.poseConf = Math.max(...Array.from(probData));
  }

  // Model 2: form check (เฉพาะ Squat)
  if (result.exercise === "Squat") {
    const hasFormFeats = meta.form_features.every((c) => c in features);
    if (hasFormFeats) {
      const formFeat  = toVector(meta.form_features, 90);
      const formInput = { float_input: new ort.Tensor("float32", formFeat, [1, meta.form_features.length]) };

      // ✅ FIX: เหมือนกัน
      const formOut   = await formSession.run(formInput);
      const fKeys     = Object.keys(formOut);
      result.form     = String(formOut[getLabelKey(fKeys)].data[0]);
      const fProbData = formOut[getProbKey(fKeys)].data as Float32Array;
      result.formConf = Math.max(...Array.from(fProbData));
      result.hasError = result.form === "Bad";
    }
  }

  return result;
}

// ─── Component ────────────────────────────────────────────────────────────────

type LoadState = "idle" | "loading" | "ready" | "error";

function PoseAnalyzer() {
  const { user } = useAuth();
  const qc = useQueryClient();

  // model state
  const [loadState, setLoadState] = useState<LoadState>("idle");
  const [loadMsg,   setLoadMsg]   = useState("");
  const yoloWorkerRef    = useRef<Worker | null>(null);
  const workerReadyRef   = useRef(false);
  const poseRfRef        = useRef<ort.InferenceSession | null>(null);
  const formClfRef       = useRef<ort.InferenceSession | null>(null);
  const metaRef          = useRef<ModelMeta | null>(null);
  const analyzingRef     = useRef(false);
  const inferBusyRef     = useRef(false);
  const inferFrameRef    = useRef(0);
  const inferIdRef       = useRef(0);
  const videoFrameCbRef  = useRef<number | undefined>(undefined);
  const prevTargetRef    = useRef<Kp[] | null>(null);
  const prevTargetTimeRef = useRef(0);
  const velocityRef      = useRef<Vel[] | null>(null);
  const lastDetectTimeRef = useRef(0);

  // session state
  const [videoSrc,   setVideoSrc]   = useState<string | null>(null);
  const [exerciseMode, setExerciseMode] = useState<ExerciseMode>("squat");
  const [analyzing,  setAnalyzing]  = useState(false);
  const [exercise,   setExercise]   = useState("—");
  const [poseConf,   setPoseConf]   = useState(0);
  const [form,       setForm]       = useState("N/A");
  const [formConf,   setFormConf]   = useState(0);
  const [formScore,  setFormScore]  = useState(0);
  const [repCount,   setRepCount]   = useState(0);
  const [latency,    setLatency]    = useState(0);
  const [frameCount, setFrameCount] = useState(0);
  const [trackStatus, setTrackStatus] = useState("");

  const videoRef       = useRef<HTMLVideoElement>(null);
  const canvasRef      = useRef<HTMLCanvasElement>(null);
  const drawRafRef     = useRef<number | undefined>(undefined);
  const isProcessing   = useRef(false);

  // display refs — updated by inference, rendered at 60fps
  const targetKpsRef    = useRef<Kp[] | null>(null);
  const displayKpsRef   = useRef<Kp[] | null>(null);
  const lastResultRef   = useRef<FrameResult | null>(null);
  const lastUiUpdateRef = useRef(0);

  // mutable refs (no re-render)
  const exerciseModeRef = useRef<ExerciseMode>("squat");
  const engineRef       = useRef(new ExerciseEngine());
  const repCtrRef       = useRef(createSquatRepCounter());
  const goodRef         = useRef(0);
  const totalRef        = useRef(0);
  const applyClassificationRef = useRef<((kps: Kp[]) => Promise<void>) | undefined>(undefined);

  useEffect(() => {
    exerciseModeRef.current = exerciseMode;
  }, [exerciseMode]);

  const handleWorkerMessage = useCallback((ev: MessageEvent) => {
    const data = ev.data as {
      type: string;
      id?: number;
      kps?: Kp[] | null;
      ms?: number;
      message?: string;
    };

    if (data.type === "error") {
      console.error("[yolo-worker]", data.message);
      inferBusyRef.current = false;
      setTrackStatus("inference error");
      return;
    }

    if (data.type !== "result") return;

    inferBusyRef.current = false;
    const vid = videoRef.current;
    const cvs = canvasRef.current;
    if (!vid || !cvs) return;

    const vw = vid.videoWidth;
    const vh = vid.videoHeight;
    if (data.kps && vw && vh) {
      const canvasKps = videoKpsToCanvas(data.kps, vw, vh, cvs.width, cvs.height);
      const now = performance.now();
      velocityRef.current = updateVelocity(prevTargetRef.current, canvasKps, prevTargetTimeRef.current, now);
      prevTargetRef.current = canvasKps;
      prevTargetTimeRef.current = now;
      lastDetectTimeRef.current = now;
      targetKpsRef.current = canvasKps;
      if (!displayKpsRef.current) displayKpsRef.current = canvasKps;

      inferFrameRef.current++;
      setLatency(data.ms ?? 0);
      setTrackStatus(`tracking · ${data.ms}ms`);

      const mode = exerciseModeRef.current;
      const runClassify =
        !exerciseUsesOnnx(mode) ||
        inferFrameRef.current % CLASSIFY_EVERY_N === 0;
      if (runClassify || !lastResultRef.current) {
        void applyClassificationRef.current?.(data.kps);
      }
    } else {
      setTrackStatus("no person detected");
    }
  }, []);

  // ── Load all models ──────────────────────────────────────────────────────────
  const loadModels = useCallback(async () => {
    if (workerReadyRef.current && yoloWorkerRef.current) {
      setLoadState("ready");
      setLoadMsg("โมเดลพร้อมใช้งาน");
      return;
    }
    setLoadState("loading");
    try {
      setLoadMsg("โหลด model_meta.json…");
      const metaRes = await fetch(`${MODEL_BASE}/model_meta_export.json`);
      if (!metaRes.ok) throw new Error("ไม่พบ model_meta_export.json ใน public/models/");
      metaRef.current = await metaRes.json();

      setLoadMsg("โหลด YOLOv8-Pose (worker)…");
      const worker = new Worker(
        new URL("../workers/pose-yolo.worker.ts", import.meta.url),
        { type: "module" },
      );
      yoloWorkerRef.current = worker;
      await new Promise<void>((resolve, reject) => {
        const timeout = window.setTimeout(() => reject(new Error("YOLO worker timeout")), 120_000);
        worker.onmessage = (ev) => {
          const data = ev.data as { type: string; message?: string };
          if (data.type === "ready") {
            workerReadyRef.current = true;
            window.clearTimeout(timeout);
            worker.onmessage = handleWorkerMessage;
            resolve();
          } else if (data.type === "error") {
            window.clearTimeout(timeout);
            reject(new Error(data.message ?? "YOLO worker failed"));
          }
        };
        worker.postMessage({
          type: "init",
          modelUrl: `${window.location.origin}${MODEL_BASE}/yolov8n-pose.onnx`,
        });
      });

      setLoadMsg("โหลด Pose Classifier ONNX…");
      poseRfRef.current = await ort.InferenceSession.create(`${MODEL_BASE}/pose_rf.onnx`, {
        executionProviders: ["wasm"],
      });

      setLoadMsg("โหลด Form Classifier ONNX…");
      formClfRef.current = await ort.InferenceSession.create(`${MODEL_BASE}/form_clf.onnx`, {
        executionProviders: ["wasm"],
      });

      setLoadState("ready");
      setLoadMsg("โมเดลพร้อมใช้งาน");
      toast.success("โหลดโมเดลสำเร็จ ✅");
    } catch (err: any) {
      setLoadState("error");
      setLoadMsg(err.message);
      toast.error("โหลดโมเดลล้มเหลว: " + err.message);
    }
  }, [handleWorkerMessage]);

  const drawLoopRef = useRef<(() => void) | undefined>(undefined);
  const captureFrameRef = useRef<() => void>(() => {});

  const syncUiFromResult = useCallback((
    exerciseName: string,
    pose: number,
    formLabel: string,
    formC: number,
    reps: number,
  ) => {
    const now = performance.now();
    if (now - lastUiUpdateRef.current >= UI_UPDATE_MS) {
      lastUiUpdateRef.current = now;
      setRepCount(reps);
      setFormScore(
        totalRef.current > 0
          ? Math.round((goodRef.current / totalRef.current) * 100)
          : 0,
      );
      setExercise(exerciseName);
      setPoseConf(pose);
      setForm(formLabel);
      setFormConf(formC);
      setFrameCount((n) => n + 1);
    }
  }, []);

  const applyClassification = useCallback(async (kps: Kp[]) => {
    const mode = exerciseModeRef.current;

    if (exerciseUsesOnnx(mode)) {
      if (!poseRfRef.current || !formClfRef.current || !metaRef.current) return;
      if (isProcessing.current) return;

      isProcessing.current = true;
      const t0 = performance.now();
      try {
        const res = await runClassifiers(
          kps, poseRfRef.current, formClfRef.current, metaRef.current,
        );
        lastResultRef.current = res;

        const knee = res.features["left_knee"] ?? res.features["right_knee"] ?? 180;
        repCtrRef.current.update(knee);
        totalRef.current++;
        if (res.form === "Good") goodRef.current++;

        syncUiFromResult(
          res.exercise,
          res.poseConf,
          res.form,
          res.formConf,
          repCtrRef.current.count,
        );
        setLatency(Math.round(performance.now() - t0));
      } catch (e) {
        console.error("[classify]", e);
      } finally {
        isProcessing.current = false;
      }
      return;
    }

    const evalRes = engineRef.current.evaluate(mode, kps, performance.now());
    lastResultRef.current = {
      exercise: evalRes.exercise,
      poseConf: evalRes.poseConf,
      form: evalRes.form,
      formConf: evalRes.formConf,
      hasError: evalRes.form === "Bad",
      features: evalRes.features,
    };

    totalRef.current++;
    if (evalRes.goodFrame) goodRef.current++;

    syncUiFromResult(
      evalRes.exercise,
      evalRes.poseConf,
      evalRes.form,
      evalRes.formConf,
      evalRes.reps,
    );
  }, [syncUiFromResult]);

  useEffect(() => {
    applyClassificationRef.current = applyClassification;
  }, [applyClassification]);

  // ── 60fps render loop ───────────────────────────────────────────────────────
  const drawLoop = useCallback(() => {
    if (!analyzingRef.current) return;

    const vid = videoRef.current;
    const cvs = canvasRef.current;
    if (vid && cvs) {
      let target = targetKpsRef.current;
      if (target) {
        const ahead = Math.min(PREDICT_MS, performance.now() - lastDetectTimeRef.current) / 1000;
        target = applyPrediction(target, velocityRef.current, ahead * 0.85);
        displayKpsRef.current = lerpKeypoints(displayKpsRef.current, target, DISPLAY_LERP);
      }

      const kps = displayKpsRef.current;
      const ctx = cvs.getContext("2d")!;
      ctx.clearRect(0, 0, cvs.width, cvs.height);

      if (kps?.some((k) => k.conf > 0.1)) {
        drawResults(
          ctx,
          kps,
          lastResultRef.current ?? {
            exercise: "—", poseConf: 0, form: "N/A", formConf: 0, hasError: false, features: {},
          },
          cvs.width,
          cvs.height,
        );
      }
    }

    drawRafRef.current = requestAnimationFrame(() => drawLoopRef.current?.());
  }, []);

  useEffect(() => { drawLoopRef.current = drawLoop; }, [drawLoop]);

  // ── Capture one video frame → YOLO worker (synced to video refresh rate) ─────
  const captureFrame = useCallback(() => {
    if (!analyzingRef.current) return;

    const vid = videoRef.current;
    const worker = yoloWorkerRef.current;
    if (!vid || !worker || !workerReadyRef.current || vid.ended) return;

    if (inferBusyRef.current || vid.readyState < 2) return;

    const vw = vid.videoWidth;
    const vh = vid.videoHeight;
    if (!vw || !vh) return;

    inferBusyRef.current = true;
    const id = ++inferIdRef.current;

    void createImageBitmap(vid)
      .then((bitmap) => {
        worker.postMessage(
          { type: "infer", id, bitmap, vw, vh, conf: CONF_THRESH },
          [bitmap],
        );
      })
      .catch((e) => {
        console.error("[capture]", e);
        inferBusyRef.current = false;
      });
  }, []);

  useEffect(() => {
    captureFrameRef.current = captureFrame;
  }, [captureFrame]);

  const scheduleVideoFrameCapture = useCallback((vid: HTMLVideoElement) => {
    type VFC = (cb: (now: number, meta: VideoFrameCallbackMetadata) => void) => number;
    const rvfc = (vid as HTMLVideoElement & { requestVideoFrameCallback?: VFC })
      .requestVideoFrameCallback;

    if (rvfc) {
      videoFrameCbRef.current = rvfc.call(vid, () => {
        if (!analyzingRef.current || vid.ended) return;
        captureFrameRef.current();
        scheduleVideoFrameCapture(vid);
      });
    } else {
      window.setTimeout(() => {
        if (!analyzingRef.current || vid.ended) return;
        captureFrameRef.current();
        scheduleVideoFrameCapture(vid);
      }, 42); // ~24fps fallback
    }
  }, []);

  // ── Draw skeleton + overlay ──────────────────────────────────────────────────
  function drawResults(
    ctx: CanvasRenderingContext2D,
    kps: Kp[],
    res: FrameResult,
    cvW: number,
    cvH: number,
  ) {
    const px = (k: Kp) => k.x;
    const py = (k: Kp) => k.y;
    const KP_MIN = 0.1;

    const color = res.form === "Good" ? "#22c55e" : res.form === "Bad" ? "#ef4444" : "#d4ff3a";

    // bounding box
    const visKps = kps.filter((k) => k.conf > KP_MIN);
    if (visKps.length > 0) {
      const xs = visKps.map((k) => k.x), ys = visKps.map((k) => k.y);
      const bx = Math.max(0, Math.min(...xs) - 12);
      const by = Math.max(0, Math.min(...ys) - 12);
      const bw = Math.min(cvW - bx, Math.max(...xs) - Math.min(...xs) + 24);
      const bh = Math.min(cvH - by, Math.max(...ys) - Math.min(...ys) + 24);
      ctx.strokeStyle = color; ctx.lineWidth = 2;
      ctx.strokeRect(bx, by, bw, bh);
      ctx.fillStyle = color;
      ctx.fillRect(bx, by - 22, Math.min(200, bw), 22);
      ctx.fillStyle = "#000"; ctx.font = "bold 11px monospace";
      ctx.fillText(
        `${res.exercise}  ${(res.poseConf * 100).toFixed(0)}%`,
        bx + 5, by - 6,
      );
    }

    // skeleton
    for (const [a, b] of SKELETON) {
      if (kps[a]?.conf > KP_MIN && kps[b]?.conf > KP_MIN) {
        ctx.strokeStyle = color; ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(px(kps[a]), py(kps[a]));
        ctx.lineTo(px(kps[b]), py(kps[b]));
        ctx.stroke();
      }
    }

    // joints
    for (const kp of kps) {
      if (kp.conf > KP_MIN) {
        ctx.fillStyle = "#fff";
        ctx.beginPath();
        ctx.arc(px(kp), py(kp), 4, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  // ── Start / Stop ─────────────────────────────────────────────────────────────
  const startAnalysis = async () => {
    if (!videoRef.current || !videoSrc || loadState !== "ready") return;
    repCtrRef.current = createSquatRepCounter();
    engineRef.current.reset();
    goodRef.current = totalRef.current = 0;
    targetKpsRef.current = null;
    displayKpsRef.current = null;
    lastResultRef.current = null;
    inferFrameRef.current = 0;
    prevTargetRef.current = null;
    velocityRef.current = null;
    lastDetectTimeRef.current = performance.now();
    setRepCount(0); setFormScore(0); setFrameCount(0);
    setTrackStatus("starting…");

    const vid = videoRef.current;
    if (vid.readyState < 2) {
      await new Promise<void>((resolve) => {
        vid.addEventListener("loadeddata", () => resolve(), { once: true });
      });
    }

    const cvs = canvasRef.current;
    if (cvs && vid.videoWidth && vid.videoHeight) {
      cvs.width = vid.videoWidth;
      cvs.height = vid.videoHeight;
    }

    vid.currentTime = 0;
    await vid.play();
    isProcessing.current = false;
    inferBusyRef.current = false;
    analyzingRef.current = true;
    setAnalyzing(true);

    requestAnimationFrame(() => {
      drawRafRef.current = requestAnimationFrame(() => drawLoopRef.current?.());
      captureFrameRef.current();
      scheduleVideoFrameCapture(vid);
    });
  };

  const stopAnalysis = useCallback(async () => {
    analyzingRef.current = false;
    if (drawRafRef.current) cancelAnimationFrame(drawRafRef.current);
    videoFrameCbRef.current = undefined;
    setTrackStatus("");
    videoRef.current?.pause();
    setAnalyzing(false);

    if (totalRef.current > 0 && user) {
      const mode = exerciseModeRef.current;
      const reps = exerciseUsesOnnx(mode)
        ? repCtrRef.current.count
        : engineRef.current.getReps(mode);
      const score = Math.round((goodRef.current / totalRef.current) * 100);
      const repLabel = mode === "plank" ? "วินาที" : "reps";
      const { error } = await supabase.from("pose_sessions").insert({
        client_id:      user.id,
        exercise_name:  exerciseLabel(mode),
        accuracy_score: score,
        feedback_json: {
          engine: exerciseUsesOnnx(mode) ? "yolov8-pose+onnx" : "yolov8-pose+rules",
          mode,
          reps,
          frames: totalRef.current,
          good_frames: goodRef.current,
        },
      });
      if (!error)
        toast.success(`บันทึกแล้ว · ${score}% form · ${reps} ${repLabel}`);
      qc.invalidateQueries({ queryKey: ["pose-history"] });
    }
  }, [user, qc]);

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (videoSrc) URL.revokeObjectURL(videoSrc);
    setVideoSrc(URL.createObjectURL(file));
    setExercise("—"); setForm("N/A"); setRepCount(0); setFormScore(0);
    goodRef.current = totalRef.current = 0;
  };

  useEffect(() => () => {
    if (videoSrc) URL.revokeObjectURL(videoSrc);
    if (drawRafRef.current) cancelAnimationFrame(drawRafRef.current);
    analyzingRef.current = false;
  }, [videoSrc]);

  useEffect(() => () => {
    yoloWorkerRef.current?.terminate();
    yoloWorkerRef.current = null;
    workerReadyRef.current = false;
  }, []);

  const { data: history } = useQuery({
    queryKey: ["pose-history", user?.id],
    queryFn: async () => {
      const { data } = await supabase
        .from("pose_sessions")
        .select("*")
        .eq("client_id", user!.id)
        .order("created_at", { ascending: false })
        .limit(10);
      return data ?? [];
    },
    enabled: !!user,
  });

  // ─── UI ───────────────────────────────────────────────────────────────────────

  const formColor =
    form === "Good" ? "text-green-500" :
    form === "Bad"  ? "text-red-500" :
    "text-muted-foreground";

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 text-xs uppercase tracking-widest text-primary">
          <Cpu className="h-3 w-3" /> YOLOv8-Pose + Random Forest · ONNX
        </div>
        <h1 className="mt-1 font-display text-4xl font-bold">Pose Analysis</h1>
        <p className="mt-2 text-muted-foreground">
          วิเคราะห์ด้วย model ที่เทรนมาจริง — จำแนกท่า + ประเมินฟอร์ม Good/Bad
        </p>
      </div>

      {/* Model loader banner */}
      {loadState !== "ready" && (
        <div className={cn(
          "rounded-xl border p-4 flex items-center justify-between",
          loadState === "error"   ? "border-destructive/50 bg-destructive/5" :
          loadState === "loading" ? "border-primary/30 bg-primary/5" :
          "border-border bg-card",
        )}>
          <div className="flex items-center gap-3">
            {loadState === "loading" ? (
              <Loader2 className="h-5 w-5 animate-spin text-primary" />
            ) : loadState === "error" ? (
              <AlertTriangle className="h-5 w-5 text-destructive" />
            ) : (
              <Cpu className="h-5 w-5 text-muted-foreground" />
            )}
            <div>
              <div className="font-semibold text-sm">
                {loadState === "idle"    && "ยังไม่ได้โหลดโมเดล"}
                {loadState === "loading" && "กำลังโหลดโมเดล…"}
                {loadState === "error"   && "โหลดโมเดลล้มเหลว"}
              </div>
              <div className="text-xs text-muted-foreground mt-0.5">{loadMsg}</div>
              {loadState === "error" && (
                <div className="text-xs text-destructive/80 mt-1">
                  วางไฟล์ .onnx ใน <code className="font-mono">public/models/</code> แล้วลองใหม่
                </div>
              )}
            </div>
          </div>
          {(loadState === "idle" || loadState === "error") && (
            <button
              onClick={loadModels}
              className="rounded-md bg-primary px-4 py-2 text-sm font-bold text-primary-foreground"
            >
              โหลดโมเดล
            </button>
          )}
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-[1fr_300px]">
        {/* Video panel */}
        <div className="rounded-xl border border-border bg-card overflow-hidden">
          <div className="relative flex aspect-video items-center justify-center bg-black">
            {videoSrc ? (
              <video
                ref={videoRef}
                src={videoSrc}
                playsInline
                muted
                preload="auto"
                className="h-full w-full object-contain"
                onEnded={stopAnalysis}
                onLoadedMetadata={(e) => {
                  const v = e.currentTarget;
                  const cvs = canvasRef.current;
                  if (cvs && v.videoWidth && v.videoHeight) {
                    cvs.width = v.videoWidth;
                    cvs.height = v.videoHeight;
                  }
                }}
              />
            ) : (
              <div className="flex flex-col items-center gap-4 p-12 text-center">
                <div className="rounded-full bg-surface-elevated p-6">
                  <FileVideo className="h-12 w-12 text-muted-foreground" />
                </div>
                <h3 className="text-xl font-bold text-white">อัปโหลดวิดีโอ</h3>
                <label className="cursor-pointer rounded-md bg-primary px-6 py-2.5 font-bold text-primary-foreground">
                  <Upload className="mr-2 inline h-4 w-4" /> เลือกไฟล์
                  <input type="file" accept="video/*" className="hidden" onChange={handleFile} />
                </label>
              </div>
            )}

            {/* Overlay canvas */}
            <canvas
              ref={canvasRef}
              width={1280}
              height={720}
              className="absolute inset-0 h-full w-full object-contain pointer-events-none"
              style={{ opacity: analyzing ? 1 : 0 }}
            />

            {/* HUD */}
            {analyzing && (
              <>
                {/* top-left */}
                <div className="absolute left-4 top-4 flex flex-col gap-1.5">
                  <div className="flex items-center gap-2 rounded-full bg-primary/90 px-3 py-1 text-[10px] font-bold text-primary-foreground">
                    <span className="h-2 w-2 animate-pulse rounded-full bg-white" />
                    YOLO LIVE
                  </div>
                  <div className="flex items-center gap-2 rounded-full bg-black/60 px-3 py-1 text-[10px] font-mono text-white backdrop-blur">
                    <Target className="h-3 w-3" /> {latency}ms · f{frameCount}
                    {trackStatus && ` · ${trackStatus}`}
                  </div>
                  {form === "Good" && (
                    <div className="flex items-center gap-2 rounded-full bg-green-500/90 px-3 py-1 text-xs font-bold text-white animate-in zoom-in">
                      <CheckCircle2 className="h-3 w-3" /> GOOD FORM
                    </div>
                  )}
                  {form === "Bad" && (
                    <div className="flex items-center gap-2 rounded-full bg-red-500/90 px-3 py-1 text-xs font-bold text-white animate-in zoom-in">
                      <AlertTriangle className="h-3 w-3" /> BAD FORM
                    </div>
                  )}
                </div>

                {/* top-right: form score */}
                <div className="absolute right-4 top-4 rounded-lg bg-background/90 px-4 py-2 text-center backdrop-blur border border-primary/20">
                  <div className="text-[10px] uppercase tracking-widest text-muted-foreground">Form Score</div>
                  <div className={cn("font-display text-3xl font-bold", formColor)}>{formScore}%</div>
                  <div className="text-[10px] text-muted-foreground mt-0.5">
                    {(formConf * 100).toFixed(0)}% conf
                  </div>
                </div>

                {/* bottom-left: reps / hold */}
                <div className="absolute left-4 bottom-16 rounded-lg bg-background/90 px-4 py-2 text-center backdrop-blur border border-primary/20">
                  <div className="text-[10px] uppercase tracking-widest text-muted-foreground">
                    {exerciseMode === "plank" ? "Hold (s)" : "Reps"}
                  </div>
                  <div className="font-display text-3xl font-bold text-primary">{repCount}</div>
                </div>

                {/* bottom-center: exercise name */}
                <div className="absolute bottom-6 left-1/2 -translate-x-1/2 rounded-lg bg-background/90 px-5 py-2 backdrop-blur border border-primary/10 text-center">
                  <div className="text-sm font-bold text-primary">{exercise}</div>
                  <div className="text-[10px] text-muted-foreground">
                    Pose conf: {(poseConf * 100).toFixed(0)}%
                  </div>
                </div>
              </>
            )}
          </div>

          {/* Controls */}
          <div className="flex items-center justify-between p-5">
            <div className="flex items-center gap-3">
              {videoSrc && !analyzing && (
                <label className="cursor-pointer rounded-md border border-border px-3 py-1.5 text-xs font-bold hover:bg-surface-elevated transition">
                  <RotateCcw className="mr-1 inline h-3 w-3" /> เปลี่ยนวิดีโอ
                  <input type="file" accept="video/*" className="hidden" onChange={handleFile} />
                </label>
              )}
            </div>
            {analyzing ? (
              <button
                onClick={stopAnalysis}
                className="flex items-center gap-2 rounded-md bg-destructive px-5 py-2.5 font-bold text-destructive-foreground"
              >
                <Square className="h-4 w-4" /> หยุด & บันทึก
              </button>
            ) : (
              <button
                onClick={startAnalysis}
                disabled={!videoSrc || loadState !== "ready"}
                className="flex items-center gap-2 rounded-md bg-primary px-5 py-2.5 font-bold text-primary-foreground disabled:opacity-40"
              >
                {loadState === "loading" ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Play className="h-4 w-4" />
                )}
                {loadState === "loading" ? "กำลังโหลด…" : "เริ่มวิเคราะห์"}
              </button>
            )}
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-4">
          {/* Exercise mode */}
          <div className="rounded-xl border border-border bg-card p-4 space-y-2">
            <div className="text-xs uppercase tracking-widest text-muted-foreground">ท่าออกกำลังกาย</div>
            <select
              value={exerciseMode}
              disabled={analyzing}
              onChange={(e) => {
                const mode = e.target.value as ExerciseMode;
                setExerciseMode(mode);
                setExercise(exerciseLabel(mode));
                setForm("N/A");
                setRepCount(0);
                setFormScore(0);
                lastResultRef.current = null;
              }}
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm font-medium disabled:opacity-50"
            >
              {EXERCISE_OPTIONS.map((o) => (
                <option key={o.id} value={o.id}>{o.label}</option>
              ))}
            </select>
            <p className="text-[10px] text-muted-foreground">
              {exerciseUsesOnnx(exerciseMode)
                ? "ใช้ ONNX classifier (Squat)"
                : "ใช้ rule-based จาก keypoints"}
            </p>
          </div>

          {/* Live stats */}
          <div className="rounded-xl border border-border bg-card p-4 space-y-3">
            <div className="text-xs uppercase tracking-widest text-muted-foreground">สถิติ Live</div>
            <Row label="ท่าที่พบ"     value={exercise} />
            <Row label="Form"          value={form} valueClass={formColor} />
            <Row label="Form Score"    value={`${formScore}%`} />
            <Row
              label={exerciseMode === "plank" ? "Hold (s)" : "Reps"}
              value={String(repCount)}
            />
            <Row label="Pose Conf"     value={`${(poseConf * 100).toFixed(1)}%`} />
            <Row label="Form Conf"     value={`${(formConf * 100).toFixed(1)}%`} />
            <Row label="Latency/frame" value={`${latency} ms`} mono />
          </div>

          {/* History */}
          <div className="rounded-xl border border-border bg-card p-4">
            <div className="mb-3 text-xs uppercase tracking-widest text-muted-foreground">ประวัติ</div>
            {history && history.length > 0 ? (
              <div className="space-y-3">
                {history.map((h: any) => (
                  <div key={h.id} className="flex items-center justify-between border-b border-border pb-2 last:border-0">
                    <div>
                      <div className="text-sm font-bold">{h.exercise_name}</div>
                      <div className="text-[10px] text-muted-foreground">
                        {new Date(h.created_at).toLocaleDateString("th-TH")}
                        {h.feedback_json?.reps != null && ` · ${h.feedback_json.reps} reps`}
                      </div>
                    </div>
                    <div className="text-sm font-bold text-primary">{h.accuracy_score}%</div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="py-4 text-center text-xs text-muted-foreground italic">ยังไม่มีประวัติ</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function Row({ label, value, valueClass, mono }: {
  label: string; value: string; valueClass?: string; mono?: boolean;
}) {
  return (
    <div className="flex justify-between text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className={cn("font-bold", mono && "font-mono text-xs", valueClass)}>{value}</span>
    </div>
  );
}
