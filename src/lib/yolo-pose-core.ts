/** Shared YOLOv8-pose preprocess/postprocess (no DOM — safe for workers). */

export const YOLO_INPUT_SIZE = 640;

export interface Kp {
  x: number;
  y: number;
  conf: number;
}

export function buildTensorFromRgba(
  data: Uint8ClampedArray,
  buffer: Float32Array | null,
): { tensor: Float32Array; buffer: Float32Array } {
  const n = YOLO_INPUT_SIZE * YOLO_INPUT_SIZE;
  const buf = buffer && buffer.length === 3 * n ? buffer : new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    buf[i] = data[i * 4] / 255;
    buf[n + i] = data[i * 4 + 1] / 255;
    buf[2 * n + i] = data[i * 4 + 2] / 255;
  }
  return { tensor: buf, buffer: buf };
}

export function postprocessYolo(
  data: Float32Array,
  dims: readonly number[],
  confThresh: number,
  scale: number,
  padX: number,
  padY: number,
  origW: number,
  origH: number,
): Kp[] | null {
  let numDet: number;
  let isHWC: boolean;

  if (dims.length === 3) {
    if (dims[1] === 56) {
      numDet = dims[2];
      isHWC = false;
    } else {
      numDet = dims[1];
      isHWC = true;
    }
  } else {
    numDet = 8400;
    isHWC = true;
  }

  let bestConf = confThresh;
  let bestOffset = -1;
  for (let d = 0; d < numDet; d++) {
    const conf = isHWC ? data[d * 56 + 4] : data[4 * numDet + d];
    if (conf > bestConf) {
      bestConf = conf;
      bestOffset = d;
    }
  }
  if (bestOffset === -1) return null;

  const det = bestOffset;
  const kps: Kp[] = [];
  for (let k = 0; k < 17; k++) {
    let px: number, py: number, pc: number;
    if (isHWC) {
      const base = det * 56 + 5 + k * 3;
      px = data[base];
      py = data[base + 1];
      pc = data[base + 2];
    } else {
      const base = (5 + k * 3) * numDet;
      px = data[base + det];
      py = data[base + numDet + det];
      pc = data[base + 2 * numDet + det];
    }
    kps.push({
      x: Math.max(0, Math.min(origW, (px - padX) / scale)),
      y: Math.max(0, Math.min(origH, (py - padY) / scale)),
      conf: pc,
    });
  }
  return kps;
}
