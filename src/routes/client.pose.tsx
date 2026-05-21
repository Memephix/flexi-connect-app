/**
 * /client/pose — YOLOv8-Pose + Random Forest + Form Classifier
 * ทำงานใน browser 100% ผ่าน onnxruntime-web
 *
 * วางไฟล์เหล่านี้ใน public/models/ ก่อน:
 *   - yolov8n-pose.onnx   (~6 MB)  — จาก ultralytics export
 *   - pose_rf.onnx         (~1 MB)  — จาก convert_models_to_onnx.py
 *   - form_clf.onnx        (~1 MB)  — จาก convert_models_to_onnx.py
 *   - model_meta_export.json        — feature/class lists
 */

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
import * as ort from "onnxruntime-web";

// ort WASM path — ใช้ CDN เดียวกับที่ vite bundle ไว้
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

export const Route = createFileRoute("/client/pose")({
  component: () => (
    <RoleGuard role="client">
      <PoseAnalyzer />
    </RoleGuard>
  ),
});

// ─── Constants ────────────────────────────────────────────────────────────────

const MODEL_BASE = "/models";
const INPUT_SIZE = 640; // YOLOv8 input

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
  form: string;       // "Good" | "Bad" | "N/A"
  formConf: number;
  hasError: boolean;
  features: Record<string, number>;
}

// ─── Geometry (ตรงกับ notebook extract_features เป๊ะ) ────────────────────────

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

/** คัดลอก logic จาก extract_features() ใน notebook เป๊ะ */
function extractFeatures(kps: Kp[]): Record<string, number> {
  const f: Record<string, number> = {};
  const ok = (...idxs: number[]) => idxs.every((i) => kps[i]?.conf > 0);
  const xy = (i: number): [number, number] => [kps[i].x, kps[i].y];

  // features เดิม
  if (ok(11, 13, 15)) f["left_knee"]  = calcAngle(xy(11), xy(13), xy(15));
  if (ok(12, 14, 16)) f["right_knee"] = calcAngle(xy(12), xy(14), xy(16));
  if (ok(5,  11, 13)) f["left_hip"]   = calcAngle(xy(5),  xy(11), xy(13));
  if (ok(6,  12, 14)) f["right_hip"]  = calcAngle(xy(6),  xy(12), xy(14));
  if ("left_knee"  in f && "right_knee" in f)
    f["knee_symmetry"] = Math.abs(f["left_knee"] - f["right_knee"]);

  // features ใหม่
  if (ok(5,  11, 13)) f["trunk_lean"]        = calcAngle(xy(5),  xy(11), xy(13));
  if (ok(13, 15, 11)) f["left_ankle_flex"]   = calcAngle(xy(13), xy(15), xy(11));
  if (ok(14, 16, 12)) f["right_ankle_flex"]  = calcAngle(xy(14), xy(16), xy(12));
  if (ok(13, 15))     f["left_knee_forward"] = kps[13].x - kps[15].x;
  if (ok(14, 16))     f["right_knee_forward"]= kps[14].x - kps[16].x;
  if ("left_hip" in f && "right_hip" in f)
    f["hip_symmetry"] = Math.abs(f["left_hip"] - f["right_hip"]);

  return f;
}

// ─── YOLOv8 Preprocessing ─────────────────────────────────────────────────────

/** แปลง video frame → Float32Array [1, 3, 640, 640] normalized [0,1] */
function preprocessFrame(
  video: HTMLVideoElement,
  tmpCanvas: HTMLCanvasElement,
): { tensor: ort.Tensor; scale: number; padX: number; padY: number } {
  const ctx = tmpCanvas.getContext("2d", { willReadFrequently: true })!;
  const vw = video.videoWidth, vh = video.videoHeight;

  // letterbox: fit vw×vh into INPUT_SIZE×INPUT_SIZE
  const scale = Math.min(INPUT_SIZE / vw, INPUT_SIZE / vh);
  const nw = Math.round(vw * scale), nh = Math.round(vh * scale);
  const padX = (INPUT_SIZE - nw) / 2, padY = (INPUT_SIZE - nh) / 2;

  tmpCanvas.width  = INPUT_SIZE;
  tmpCanvas.height = INPUT_SIZE;

  ctx.fillStyle = "#808080"; // grey pad
  ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  ctx.drawImage(video, padX, padY, nw, nh);

  const imgData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  const { data } = imgData;
  const n = INPUT_SIZE * INPUT_SIZE;

  // RGBA → CHW float32 /255
  const float32 = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    float32[i]         = data[i * 4]     / 255; // R
    float32[n + i]     = data[i * 4 + 1] / 255; // G
    float32[2 * n + i] = data[i * 4 + 2] / 255; // B
  }

  return {
    tensor: new ort.Tensor("float32", float32, [1, 3, INPUT_SIZE, INPUT_SIZE]),
    scale,
    padX,
    padY,
  };
}

// ─── YOLOv8 Postprocessing ────────────────────────────────────────────────────

/** NMS — เลือก detection ที่ conf สูงสุด (ไม่ทำ full NMS เพราะ pose เราสนใจแค่คนเดียว) */
function postprocessYolo(
  output: ort.Tensor,   // [1, 56, 8400]
  confThresh: number,
  scale: number,
  padX: number,
  padY: number,
  origW: number,
  origH: number,
): Kp[] | null {
  const data = output.data as Float32Array;
  // shape [1, 56, 8400] → dims[1]=56, dims[2]=8400
  const numDet    = 8400;
  const stride    = 56;   // 4 bbox + 1 conf + 17*3 kps

  let bestConf = confThresh;
  let bestOffset = -1;

  // หา detection ที่ confidence สูงที่สุด
  for (let d = 0; d < numDet; d++) {
    const conf = data[4 * numDet + d]; // row 4 = objectness
    if (conf > bestConf) {
      bestConf   = conf;
      bestOffset = d;
    }
  }

  if (bestOffset === -1) return null;

  const d = bestOffset;
  const kps: Kp[] = [];

  for (let k = 0; k < 17; k++) {
    const base = (5 + k * 3) * numDet;
    // undo letterbox
    const px = (data[base + d]           - padX) / scale;
    const py = (data[(base + numDet) + d] - padY) / scale;
    const pc =  data[(base + 2 * numDet) + d];

    kps.push({
      x: Math.max(0, Math.min(origW, px)),
      y: Math.max(0, Math.min(origH, py)),
      conf: pc,
    });
  }

  return kps;
}

// ─── Rep Counter (ตรง notebook) ───────────────────────────────────────────────

class RepCounter {
  count = 0;
  stage: "up" | "down" = "up";
  constructor(readonly upThresh = 150, readonly downThresh = 100) {}
  update(kneeAngle: number): number {
    if (kneeAngle > this.upThresh) this.stage = "up";
    else if (kneeAngle < this.downThresh && this.stage === "up") {
      this.stage = "down";
      this.count++;
    }
    return this.count;
  }
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

  // Model 1: classify pose
  const hasPoseFeats = meta.pose_features.every((c) => c in features);
  if (hasPoseFeats) {
    const poseFeat = toVector(meta.pose_features, 180);
    const poseInput = { float_input: new ort.Tensor("float32", poseFeat, [1, meta.pose_features.length]) };
    const [labels, probs] = await poseSession.run(poseInput);
    const labelArr = labels.data as BigInt64Array | Int64Array | string[] | any;
    result.exercise = String(labelArr[0]);
    const probArr   = probs.data as Float32Array;
    result.poseConf = Math.max(...Array.from(probArr));
  }

  // Model 2: form check (เฉพาะ Squat)
  if (result.exercise === "Squat") {
    const hasFormFeats = meta.form_features.every((c) => c in features);
    if (hasFormFeats) {
      const formFeat  = toVector(meta.form_features, 90);
      const formInput = { float_input: new ort.Tensor("float32", formFeat, [1, meta.form_features.length]) };
      const [fLabels, fProbs] = await formSession.run(formInput);
      const fLabelArr = fLabels.data as any;
      result.form     = String(fLabelArr[0]);
      const fProbArr  = fProbs.data as Float32Array;
      result.formConf = Math.max(...Array.from(fProbArr));
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
  const yoloRef      = useRef<ort.InferenceSession | null>(null);
  const poseRfRef    = useRef<ort.InferenceSession | null>(null);
  const formClfRef   = useRef<ort.InferenceSession | null>(null);
  const metaRef      = useRef<ModelMeta | null>(null);
  const tmpCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // session state
  const [videoSrc,      setVideoSrc]      = useState<string | null>(null);
  const [analyzing,     setAnalyzing]     = useState(false);
  const [exercise,      setExercise]      = useState("—");
  const [poseConf,      setPoseConf]      = useState(0);
  const [form,          setForm]          = useState("N/A");
  const [formConf,      setFormConf]      = useState(0);
  const [formScore,     setFormScore]     = useState(0);
  const [repCount,      setRepCount]      = useState(0);
  const [latency,       setLatency]       = useState(0);
  const [frameCount,    setFrameCount]    = useState(0);

  const videoRef  = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef    = useRef<number>();

  // mutable refs (no re-render)
  const repCtrRef    = useRef(new RepCounter());
  const maxScoreRef  = useRef(0);
  const goodRef      = useRef(0);
  const totalRef     = useRef(0);

  // ── Load all models ──────────────────────────────────────────────────────────
  const loadModels = useCallback(async () => {
    setLoadState("loading");
    try {
      setLoadMsg("โหลด model_meta.json…");
      const metaRes = await fetch(`${MODEL_BASE}/model_meta_export.json`);
      if (!metaRes.ok) throw new Error("ไม่พบ model_meta_export.json ใน public/models/");
      metaRef.current = await metaRes.json();

      setLoadMsg("โหลด YOLOv8-Pose ONNX (~6 MB)…");
      yoloRef.current = await ort.InferenceSession.create(`${MODEL_BASE}/yolov8n-pose.onnx`, {
        executionProviders: ["wasm"],
      });

      setLoadMsg("โหลด Pose Classifier ONNX…");
      poseRfRef.current = await ort.InferenceSession.create(`${MODEL_BASE}/pose_rf.onnx`, {
        executionProviders: ["wasm"],
      });

      setLoadMsg("โหลด Form Classifier ONNX…");
      formClfRef.current = await ort.InferenceSession.create(`${MODEL_BASE}/form_clf.onnx`, {
        executionProviders: ["wasm"],
      });

      tmpCanvasRef.current = document.createElement("canvas");

      setLoadState("ready");
      setLoadMsg("โมเดลพร้อมใช้งาน");
      toast.success("โหลดโมเดลสำเร็จ ✅");
    } catch (err: any) {
      setLoadState("error");
      setLoadMsg(err.message);
      toast.error("โหลดโมเดลล้มเหลว: " + err.message);
    }
  }, []);

  // ── Process one frame ────────────────────────────────────────────────────────
  const processFrame = useCallback(async () => {
    const vid = videoRef.current;
    const cvs = canvasRef.current;
    if (!vid || !cvs || vid.paused || vid.ended) return;
    if (!yoloRef.current || !poseRfRef.current || !formClfRef.current || !metaRef.current) return;

    const t0 = performance.now();

    // 1. Preprocess
    const { tensor, scale, padX, padY } = preprocessFrame(vid, tmpCanvasRef.current!);

    // 2. YOLOv8 inference
    const yoloOut = await yoloRef.current.run({ images: tensor });
    const output  = yoloOut[Object.keys(yoloOut)[0]]; // [1, 56, 8400]

    const kps = postprocessYolo(output, 0.4, scale, padX, padY, vid.videoWidth, vid.videoHeight);

    const ctx = cvs.getContext("2d")!;
    ctx.clearRect(0, 0, cvs.width, cvs.height);

    if (kps) {
      // 3. Classify
      const res = await runClassifiers(kps, poseRfRef.current, formClfRef.current, metaRef.current);

      // 4. Rep counting
      const kneeAngle = res.features["left_knee"] ?? 180;
      const reps = repCtrRef.current.update(kneeAngle);
      setRepCount(reps);

      // 5. Running form score
      totalRef.current++;
      if (res.form === "Good") goodRef.current++;
      const runningScore = totalRef.current > 0
        ? Math.round((goodRef.current / totalRef.current) * 100)
        : 0;
      setFormScore(runningScore);

      // 6. Update UI
      setExercise(res.exercise);
      setPoseConf(res.poseConf);
      setForm(res.form);
      setFormConf(res.formConf);
      setFrameCount((n) => n + 1);

      // 7. Draw on canvas
      drawResults(ctx, kps, res, vid.videoWidth, vid.videoHeight, cvs.width, cvs.height);
    }

    setLatency(Math.round(performance.now() - t0));
    rafRef.current = requestAnimationFrame(processFrame);
  }, []);

  // ── Draw skeleton + overlay ──────────────────────────────────────────────────
  function drawResults(
    ctx: CanvasRenderingContext2D,
    kps: Kp[],
    res: FrameResult,
    origW: number, origH: number,
    cvW: number, cvH: number,
  ) {
    const sx = cvW / origW, sy = cvH / origH;
    const px = (k: Kp) => k.x * sx, py = (k: Kp) => k.y * sy;

    const color = res.form === "Good" ? "#22c55e" : res.form === "Bad" ? "#ef4444" : "#d4ff3a";

    // bounding box
    const visKps = kps.filter((k) => k.conf > 0.3);
    if (visKps.length > 0) {
      const xs = visKps.map((k) => k.x * sx), ys = visKps.map((k) => k.y * sy);
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
      if (kps[a]?.conf > 0.3 && kps[b]?.conf > 0.3) {
        ctx.strokeStyle = color; ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(px(kps[a]), py(kps[a]));
        ctx.lineTo(px(kps[b]), py(kps[b]));
        ctx.stroke();
      }
    }

    // joints
    for (const kp of kps) {
      if (kp.conf > 0.3) {
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
    repCtrRef.current = new RepCounter();
    goodRef.current = totalRef.current = 0;
    maxScoreRef.current = 0;
    setRepCount(0); setFormScore(0); setFrameCount(0);
    videoRef.current.currentTime = 0;
    await videoRef.current.play();
    setAnalyzing(true);
    rafRef.current = requestAnimationFrame(processFrame);
  };

  const stopAnalysis = useCallback(async () => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    videoRef.current?.pause();
    setAnalyzing(false);

    if (totalRef.current > 0 && user) {
      const score = Math.round((goodRef.current / totalRef.current) * 100);
      const { error } = await supabase.from("pose_sessions").insert({
        client_id:      user.id,
        exercise_name:  exercise,
        accuracy_score: score,
        feedback_json: {
          engine: "yolov8-pose+onnx",
          reps: repCtrRef.current.count,
          frames: totalRef.current,
          good_frames: goodRef.current,
        },
      });
      if (!error)
        toast.success(`บันทึกแล้ว · ${score}% form · ${repCtrRef.current.count} reps`);
      qc.invalidateQueries({ queryKey: ["pose-history"] });
    }
  }, [exercise, user, qc]);

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
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
  }, [videoSrc]);

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
          <Cpu className="h-3 w-3" /> YOLOv8-Pose + Random Forest · ONNX Runtime
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
          "border-border bg-card"
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
                playsInline muted
                className="h-full w-full object-contain"
                onEnded={stopAnalysis}
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
              width={1280} height={720}
              className={cn(
                "absolute inset-0 h-full w-full pointer-events-none",
                !analyzing && "hidden",
              )}
            />

            {/* HUD */}
            {analyzing && (
              <>
                {/* top-left */}
                <div className="absolute left-4 top-4 flex flex-col gap-1.5">
                  <div className="flex items-center gap-2 rounded-full bg-primary/90 px-3 py-1 text-[10px] font-bold text-primary-foreground">
                    <span className="h-2 w-2 animate-pulse rounded-full bg-white" />
                    YOLOv8 + RF LIVE
                  </div>
                  <div className="flex items-center gap-2 rounded-full bg-black/60 px-3 py-1 text-[10px] font-mono text-white backdrop-blur">
                    <Target className="h-3 w-3" /> {latency}ms · f{frameCount}
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

                {/* bottom-left: reps */}
                <div className="absolute left-4 bottom-16 rounded-lg bg-background/90 px-4 py-2 text-center backdrop-blur border border-primary/20">
                  <div className="text-[10px] uppercase tracking-widest text-muted-foreground">Reps</div>
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
          {/* Live stats */}
          <div className="rounded-xl border border-border bg-card p-4 space-y-3">
            <div className="text-xs uppercase tracking-widest text-muted-foreground">สถิติ Live</div>
            <Row label="ท่าที่พบ"       value={exercise} />
            <Row label="Form"            value={form} valueClass={formColor} />
            <Row label="Form Score"      value={`${formScore}%`} />
            <Row label="Reps"            value={String(repCount)} />
            <Row label="Pose Conf"       value={`${(poseConf * 100).toFixed(1)}%`} />
            <Row label="Form Conf"       value={`${(formConf * 100).toFixed(1)}%`} />
            <Row label="Latency/frame"   value={`${latency} ms`} mono />
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
