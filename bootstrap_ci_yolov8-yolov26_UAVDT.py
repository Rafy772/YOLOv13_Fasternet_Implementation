"""
Bootstrap 95% confidence intervals for YOLO detection metrics (memory-safe).

Memory design (this is the part that matters):
  - Inference runs ONE image at a time, so only a single decoded image is ever
    in RAM. (Passing the whole image list to predict can materialize every
    Results object -- each holding the full image -- at once, which overruns RAM.)
  - Per image we keep only tiny arrays: the correctness matrix tp (n_det x 10),
    confidence, predicted class, image index, and the GT-class counts.
  - The detections are concatenated and sorted by confidence ONCE.
  - Each bootstrap resample is just an integer weight per image; recomputing the
    metrics under those weights is mathematically identical to duplicating the
    drawn images but uses a flat, tiny amount of memory.
  - A live RSS monitor prints memory every PROGRESS_EVERY images and ABORTS if it
    crosses MAX_RSS_GB, so the script can never freeze the machine again.

Self-contained: IoU, matching and COCO-style AP (101-pt interpolation, IoU
0.50:0.95) are implemented here; only YOLO(...) is used, for inference.

Outputs BOTH the class-mean metrics (class='all') and per-class metrics, each
with a bootstrap 95% CI. Per-class bootstrapping is free: the per-class P/R/F1/AP
values are already computed for the mean, so they are just carried through the
same resampling loop. Class names are read from `names` in the data YAML.

Inference time (ms per image, pure forward pass as reported by Ultralytics in
`r.speed['inference']`) is recorded once during that same single inference pass
and carried through the identical resample loop, so it gets a bootstrap CI too.
NOTE on interpretation: resampling images captures image-to-image variation in
forward time (small at fixed imgsz, so this CI is usually narrow). It does NOT
capture run-to-run system noise (GPU clocks, thermals, other load). A WARMUP
pass is run first so the timed loop measures steady state, not CUDA/cuDNN init.

Run once: fill MODELS + DATA_YAML, then `python bootstrap_ci.py`.
"""

import os
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

try:
    import torch
except Exception:
    torch = None

_trapz = getattr(np, "trapezoid", np.trapz)

# -------------------------------- config --------------------------------
MODELS = {
    'yolov26': '/home/user129/yolov13/Paper_Hasil_Ujicoba/IKOMTI/UAVDT_YOLOv26n_640_Baseline/train-2/weights/best.pt',
    'yolov13': '/home/user129/yolov13/Paper_Hasil_Ujicoba/IKOMTI/UAVDT_YOLOV13n_640_Baseline/weights/best.pt',
    'yolov12': '/home/user129/yolov13/Paper_Hasil_Ujicoba/IKOMTI/UAVDT_YOLOv12n_640_Baseline/train177/weights/best.pt',
    'yolov11': '/home/user129/yolov13/Paper_Hasil_Ujicoba/IKOMTI/UAVDT_YOLOv11n_640_Baseline/train178/weights/best.pt',
    'yolov10': '/home/user129/yolov13/Paper_Hasil_Ujicoba/IKOMTI/UAVDT_YOLOv10n_64_Baseline/train179/weights/best.pt',
    'yolov9': '/home/user129/yolov13/Paper_Hasil_Ujicoba/IKOMTI/UAVDT_YOLOv9n_640_Baseline/train180/weights/best.pt',
    'yolov8': '/home/user129/yolov13/Paper_Hasil_Ujicoba/IKOMTI/UAVDT_YOLOv8n_640_Baseline/train181/weights/best.pt',
 # 'Baseline':  '/path/to/baseline/weights/best.pt',
    # ...
}
DATA_YAML = 'ultralytics/cfg/datasets/UAVDT.yaml'  # val split is read from here

IMGSZ    = 640
CONF     = 0.01      # detections kept for the PR curve (0.01 vs val's 0.001 changes mAP <~0.002)
IOU_NMS  = 0.7
MAX_DET  = 300
DEVICE   = '0'       # GPU 0  ('cpu' for CPU)

WARMUP   = 5         # discarded inferences before timing (CUDA init + cuDNN autotune)

N_BOOT   = 1000
ALPHA    = 0.05      # 95% CI
SEED     = 0

PROGRESS_EVERY = 200 # print images-done + RSS this often
MAX_RSS_GB     = 24  # abort if memory crosses this (keep below your physical RAM)

IOUV     = np.linspace(0.5, 0.95, 10)
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
OUT_CSV  = 'bootstrap_ci.csv'

METRICS  = ['P', 'R', 'F1', 'mAP50', 'mAP50_95']  # per-class + class-mean metrics
TIME_METRIC = 'infer_ms'                           # global (not per-class) metric
# ------------------------------------------------------------------------


def mem_gb():
    try:
        for line in open('/proc/self/status'):
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024 / 1024   # kB -> GB
    except Exception:
        pass
    return float('nan')


def resolve_val_images(yaml_path):
    cfg = yaml.safe_load(Path(yaml_path).read_text())
    root = Path(cfg.get('path', '.')).expanduser().resolve()
    entry = cfg['val']
    src = Path(entry) if Path(entry).is_absolute() else (root / entry)
    src = src.resolve()
    if src.is_dir():
        imgs = [p for p in src.rglob('*') if p.suffix.lower() in IMG_EXTS]
    else:
        imgs = [Path(l.strip()) for l in src.read_text().splitlines() if l.strip()]
    return sorted({p.resolve() for p in imgs})


def load_class_names(yaml_path):
    """Return {class_id: name} from the data YAML. Handles dict or list `names`."""
    cfg = yaml.safe_load(Path(yaml_path).read_text())
    names = cfg.get('names', {})
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(v) for i, v in enumerate(names)}


def img2label(img_path):
    p = str(img_path)
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'
    if sa in p:
        p = sb.join(p.rsplit(sa, 1))
    return Path(p).with_suffix('.txt')


def load_gt(img_path, w, h):
    lp = img2label(img_path)
    if not lp.exists():
        return np.zeros((0, 4)), np.zeros((0,), dtype=int)
    boxes, cls = [], []
    for line in lp.read_text().splitlines():
        t = line.split()
        if len(t) < 5:
            continue
        c, cx, cy, bw, bh = int(float(t[0])), *map(float, t[1:5])
        boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                      (cx + bw / 2) * w, (cy + bh / 2) * h])
        cls.append(c)
    return np.array(boxes, dtype=float).reshape(-1, 4), np.array(cls, dtype=int)


def box_iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (br - tl).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-16)


def match(gt_boxes, gt_cls, det_boxes, det_cls):
    tp = np.zeros((len(det_cls), len(IOUV)), dtype=bool)
    if len(det_cls) == 0 or len(gt_cls) == 0:
        return tp
    iou = box_iou(gt_boxes, det_boxes) * (gt_cls[:, None] == det_cls[None, :])
    for ti, thr in enumerate(IOUV):
        gi, di = np.where(iou >= thr)
        if len(gi):
            order = iou[gi, di].argsort()[::-1]
            gi, di = gi[order], di[order]
            u = np.unique(di, return_index=True)[1]
            gi, di = gi[u], di[u]
            di = di[np.unique(gi, return_index=True)[1]]
            tp[di, ti] = True
    return tp


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return _trapz(np.interp(x, mrec, mpre), x)


def collect(model, img_list):
    """Inference ONE image at a time. Returns confidence-sorted TP, PCLS, IMG,
    a per-image GT-count matrix GTC, the class list, and TIMES -- a per-image
    array (length N, in image order, NOT sorted with detections) of forward-pass
    milliseconds. TIMES lines up with the per-image bootstrap weight vector."""
    tps, confs, pclss, imgis, gts = [], [], [], [], []
    times = []
    N = len(img_list)
    print(f'  inference start  RSS={mem_gb():.1f} GB', flush=True)

    # Warmup so the timed loop measures steady state. The first few calls include
    # lazy CUDA init and cuDNN autotuning and would otherwise inflate the mean.
    if N and WARMUP:
        for _ in range(WARMUP):
            model.predict(str(img_list[0]), imgsz=IMGSZ, conf=CONF, iou=IOU_NMS,
                          max_det=MAX_DET, device=DEVICE, verbose=False)
        if torch is not None and str(DEVICE) != 'cpu':
            torch.cuda.synchronize()
        gc.collect()
        print(f'  warmup done ({WARMUP})  RSS={mem_gb():.1f} GB', flush=True)

    for j, p in enumerate(img_list):
        r = model.predict(str(p), imgsz=IMGSZ, conf=CONF, iou=IOU_NMS,
                          max_det=MAX_DET, device=DEVICE, verbose=False)[0]
        times.append(float(r.speed['inference']))  # ms, pure forward pass (CUDA-synced by Ultralytics)
        h, w = r.orig_shape
        db = r.boxes.xyxy.cpu().numpy()
        dc = r.boxes.conf.cpu().numpy().astype(np.float32)
        dk = r.boxes.cls.cpu().numpy().astype(np.int16)
        gb, gk = load_gt(r.path, w, h)
        tps.append(match(gb, gk, db, dk))          # boxes consumed here, never stored
        confs.append(dc); pclss.append(dk)
        imgis.append(np.full(len(dc), j, dtype=np.int32)); gts.append(gk)
        del r, db, dc, dk, gb

        if (j + 1) % PROGRESS_EVERY == 0:
            gc.collect()
            if torch is not None and str(DEVICE) != 'cpu':
                torch.cuda.empty_cache()
            rss = mem_gb()
            print(f'  {j+1}/{N} images  RSS={rss:.1f} GB', flush=True)
            if rss > MAX_RSS_GB:
                raise MemoryError(
                    f'RSS hit {rss:.1f} GB at image {j+1} (limit {MAX_RSS_GB}). '
                    f'Aborting before the machine freezes. If RSS was climbing '
                    f'steadily, the leak is in inference, not the bootstrap.')

    TP   = np.concatenate(tps) if tps else np.zeros((0, len(IOUV)), bool)
    CONF_ = np.concatenate(confs) if confs else np.zeros((0,), np.float32)
    PCLS = np.concatenate(pclss) if pclss else np.zeros((0,), np.int16)
    IMG  = np.concatenate(imgis) if imgis else np.zeros((0,), np.int32)
    TIMES = np.asarray(times, dtype=np.float64)    # per-image, image order
    del tps, confs, pclss, imgis
    gc.collect()

    order = np.argsort(-CONF_)
    TP, PCLS, IMG = TP[order], PCLS[order], IMG[order]   # NB: TIMES stays image-ordered
    print(f'  inference done   detections={len(TP)}  '
          f'mean {TIMES.mean():.2f} ms/img  RSS={mem_gb():.1f} GB', flush=True)

    all_gt = np.concatenate([g for g in gts if len(g)]) if any(len(g) for g in gts) else np.array([0])
    classes = np.unique(all_gt)
    cidx = {int(c): i for i, c in enumerate(classes)}
    GTC = np.zeros((len(img_list), len(classes)), dtype=np.int32)
    for j, g in enumerate(gts):
        for c in g:
            GTC[j, cidx[int(c)]] += 1
    return TP, PCLS, IMG, GTC, classes, TIMES


def weighted_metrics(w, TP, PCLS, IMG, GTC, classes, TIMES):
    """Flat vector of length 5 + 5*len(classes) + 1:
      [class-mean P,R,F1,mAP50,mAP50_95,
       then per-class P,R,F1,mAP50,mAP50_95 ...,
       then global infer_ms].
    The per-class values were already computed for the mean, so returning them
    adds no work -- the bootstrap loop then gives every class its own CI. The
    trailing infer_ms is the mean forward time over the resampled images:
    (w * TIMES).sum() / w.sum() -- identical to averaging the drawn images."""
    det_w = w[IMG]
    nthr = TP.shape[1]
    ap = np.zeros((len(classes), nthr))
    p_c = np.zeros(len(classes)); r_c = np.zeros(len(classes)); f_c = np.zeros(len(classes))
    for ci in range(len(classes)):
        n_gt = float((w * GTC[:, ci]).sum())
        m = PCLS == classes[ci]
        if n_gt == 0 or not m.any():
            continue
        dw = det_w[m].astype(np.int32)[:, None]
        tpc = (TP[m] * dw).cumsum(0)
        cnt = dw.cumsum(0)
        recall = tpc / n_gt
        precision = tpc / np.maximum(cnt, 1)
        for ti in range(nthr):
            ap[ci, ti] = compute_ap(recall[:, ti], precision[:, ti])
        f1 = 2 * precision[:, 0] * recall[:, 0] / (precision[:, 0] + recall[:, 0] + 1e-16)
        k = f1.argmax()
        p_c[ci], r_c[ci], f_c[ci] = precision[k, 0], recall[k, 0], f1[k]

    ap50_c = ap[:, 0]        # per-class mAP50
    ap_c   = ap.mean(1)      # per-class mAP50-95
    mean = np.array([p_c.mean(), r_c.mean(), f_c.mean(), ap50_c.mean(), ap.mean()])
    per_class = np.stack([p_c, r_c, f_c, ap50_c, ap_c], axis=1).ravel()  # class-major

    wsum = float(w.sum())
    infer_ms = float((w * TIMES).sum() / wsum) if wsum else float('nan')
    return np.concatenate([mean, per_class, [infer_ms]])


def main():
    imgs = resolve_val_images(DATA_YAML)
    names = load_class_names(DATA_YAML)
    n = len(imgs)
    print(f'Validation images: {n}   (start RSS={mem_gb():.1f} GB)')
    rng = np.random.default_rng(SEED)

    rows = []
    for label, ckpt in MODELS.items():
        print(f'\n===== {label} =====')
        TP, PCLS, IMG, GTC, classes, TIMES = collect(YOLO(ckpt), imgs)
        class_labels = [names.get(int(c), f'class_{int(c)}') for c in classes]

        # column meta aligned with the flat vector: mean block, per-class blocks,
        # then the single global timing metric appended last.
        columns = [('all', me) for me in METRICS]
        for lab in class_labels:
            columns += [(lab, me) for me in METRICS]
        columns += [('all', TIME_METRIC)]

        point = weighted_metrics(np.ones(n, np.int32), TP, PCLS, IMG, GTC, classes, TIMES)
        boot = np.array([
            weighted_metrics(np.bincount(rng.integers(0, n, n), minlength=n).astype(np.int32),
                             TP, PCLS, IMG, GTC, classes, TIMES)
            for _ in range(N_BOOT)])
        lo = np.percentile(boot, 100 * ALPHA / 2, axis=0)
        hi = np.percentile(boot, 100 * (1 - ALPHA / 2), axis=0)

        print(f'  [mean over classes]')
        print(f'  {"metric":12s}{"point":>9s}{"95% CI":>22s}')
        for i, (cls, me) in enumerate(columns):
            if cls == 'all' and me in METRICS:
                print(f'  {me:12s}{point[i]:9.4f}   [{lo[i]:.4f}, {hi[i]:.4f}]')

        print(f'  [per class]')
        print(f'  {"class":16s}{"metric":10s}{"point":>9s}{"95% CI":>22s}')
        for i, (cls, me) in enumerate(columns):
            if cls != 'all':
                print(f'  {cls:16s}{me:10s}{point[i]:9.4f}   [{lo[i]:.4f}, {hi[i]:.4f}]')

        # timing (global). FPS is a monotonic transform of ms, so its percentile
        # CI is the reciprocal of the ms CI with endpoints swapped (identical up
        # to sub-percentile interpolation; add FPS to the vector if you want it
        # bootstrapped directly).
        ti = columns.index(('all', TIME_METRIC))
        ms_pt, ms_lo, ms_hi = float(point[ti]), float(lo[ti]), float(hi[ti])
        print(f'  [timing]')
        print(f'  {"inference ms/img":18s}{ms_pt:9.3f}   [{ms_lo:.3f}, {ms_hi:.3f}]')
        if ms_pt > 0 and ms_lo > 0 and ms_hi > 0:
            print(f'  {"FPS":18s}{1000.0/ms_pt:9.1f}   '
                  f'[{1000.0/ms_hi:.1f}, {1000.0/ms_lo:.1f}]')

        for i, (cls, me) in enumerate(columns):
            rows.append({'model': label, 'class': cls, 'metric': me,
                         'point': round(float(point[i]), 4),
                         'ci_low': round(float(lo[i]), 4),
                         'ci_high': round(float(hi[i]), 4)})

        del TP, PCLS, IMG, GTC, TIMES
        gc.collect()

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f'\nSaved -> {OUT_CSV}   (end RSS={mem_gb():.1f} GB)')


if __name__ == '__main__':
    main()