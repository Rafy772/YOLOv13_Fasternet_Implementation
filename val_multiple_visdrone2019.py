"""
Per-image AP50 comparison across N YOLO weights.

Runs every model over the validation folder, computes per-image AP50 for each,
and writes a grid of annotated panels for the TOP_K images with the largest
disagreement between models.

Notes:
  - mAP50 is defined over a dataset. Here we compute per-image AP50: greedy
    IoU-0.5 matching, 101-point interpolated AP, averaged over the classes
    present in that image's ground truth.
  - EVAL_CONF == DRAW_CONF by default so the AP50 in each header matches the
    boxes you can see. Set EVAL_CONF=0.001 for protocol-faithful mAP numbers,
    but then the header TP/FP/FN will disagree with AP50: the detection
    earning the AP may sit below DRAW_CONF and never get drawn.
  - DRAW_ONLY_GT_CLASSES hides predictions whose class is absent from that
    image's ground truth. Those predictions cannot affect the image's AP50,
    so drawing them only invites confusion.
  - Inference uses a .txt manifest as the predict source. Passing a list of
    paths makes ultralytics decode every image up front (LoadPilAndNumpy),
    which OOMs on large val sets even with stream=True.
"""

import csv
import gc
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
ROOT = Path('/home/rafy/yolov13/Paper_Hasil_Ujicoba/Visdrone2019_Hasil Percobaan')

WEIGHTS = {
    'yolov13_baseline':            ROOT / 'Baseline_yolov13n_640/train18/weights/best.pt',
    'yolov13_Stride 1':            ROOT / 'CBAM_yolov13n_640/weights/best.pt',
    'yolov13_Kernel 5':            ROOT / 'yolov13n_Kernel 5_640/train107/weights/best.pt',
    'yolov13_P2':                  ROOT / 'P2_yolov13n_640/train14/weights/best.pt',
    'yolov13_Partial Convolution': ROOT / 'Partial Convolution_yolov13n_640/train212/weights/best.pt',
    'yolov13_S1 + K5 + P2': ROOT / 'Semua Kombinasi_yolov13n_640/weights/best.pt',
    'yolov13_Semua Perubahan':     ROOT / 'Semua Kombinasi Partial Convolution_yolov13n_640/train245/weights/best.pt',
}


REFERENCE = 'yolov13_baseline'   # the model everything is compared against
RANK_BY = 'spread'               # 'spread' (max-min across all models)
                                 #   or a model key -> that model minus REFERENCE
SORT_BY = 'abs'                  # 'abs' | 'a_wins' (rank value high) | 'b_wins' (low)

# vvv still points at Dolphin14k -- point this at the VisDrone val split
VAL_IMAGES_DIR = '/home/rafy/yolov13/VisDrone2019/valid/images'
VAL_LABELS_DIR = '/home/rafy/yolov13/VisDrone2019/valid/labels'            # None -> replace '/images' with '/labels'
OUTPUT_DIR = '/home/rafy/yolov13/ap50_diff_analysis'

IMGSZ = 640
DEVICE = 0
DRAW_CONF = 0.25                 # detections drawn on the output images
EVAL_CONF = DRAW_CONF            # set to 0.001 for standard mAP protocol
NMS_IOU = 0.7
MAX_DET = 300
MATCH_IOU = 0.5

TOP_K = 500
SKIP_EMPTY_GT = True             # images with no GT have undefined AP -> skip

DRAW_ONLY_GT_CLASSES = True      # hide predictions of classes absent from GT
DRAW_GT_LABELS = True            # class name under each GT box
GRID_COLS = 3                    # panels per row
PANEL_MAX_W = 1100               # downscale each panel to this width
FONT_SCALE = 1.4                 # bump if text is still too small

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

# BGR
C_GT_MATCHED = (120, 220, 120)
C_GT_MISSED = (0, 215, 255)
C_TP = (255, 160, 0)
C_FP = (60, 60, 255)
C_BG = (32, 32, 32)


# ----------------------------------------------------------------------------
# geometry / metrics
# ----------------------------------------------------------------------------
def box_iou(a, b):
    """a: (N,4) xyxy, b: (M,4) xyxy -> (N,M)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


def greedy_match(pred_boxes, pred_scores, gt_boxes, iou_thr=MATCH_IOU):
    """Returns (tp_flags per pred, matched_gt flags). Preds assumed unsorted."""
    n_p, n_g = len(pred_boxes), len(gt_boxes)
    tp = np.zeros(n_p, dtype=bool)
    matched = np.zeros(n_g, dtype=bool)
    if n_p == 0 or n_g == 0:
        return tp, matched

    order = np.argsort(-pred_scores)
    ious = box_iou(pred_boxes, gt_boxes)
    for i in order:
        best_j, best_iou = -1, iou_thr
        for j in range(n_g):
            if matched[j]:
                continue
            if ious[i, j] >= best_iou:
                best_iou, best_j = ious[i, j], j
        if best_j >= 0:
            matched[best_j] = True
            tp[i] = True
    return tp, matched


def compute_ap(recall, precision):
    """101-point interpolated AP (COCO style)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapz(np.interp(x, mrec, mpre), x))


def ap50_single_class(pred_boxes, pred_scores, gt_boxes):
    n_gt = len(gt_boxes)
    if n_gt == 0:
        return None                       # class not in GT -> excluded from mean
    if len(pred_boxes) == 0:
        return 0.0

    order = np.argsort(-pred_scores)
    pb, ps = pred_boxes[order], pred_scores[order]
    tp, _ = greedy_match(pb, ps, gt_boxes)

    tpc = np.cumsum(tp.astype(np.float64))
    fpc = np.cumsum((~tp).astype(np.float64))
    recall = tpc / n_gt
    precision = tpc / np.clip(tpc + fpc, 1e-9, None)
    return compute_ap(recall, precision)


def image_ap50(pred_boxes, pred_scores, pred_cls, gt_boxes, gt_cls):
    """Mean AP50 over the classes present in this image's ground truth."""
    classes = np.unique(gt_cls).astype(int)
    if len(classes) == 0:
        return None
    aps = []
    for c in classes:
        pm, gm = pred_cls == c, gt_cls == c
        ap = ap50_single_class(pred_boxes[pm], pred_scores[pm], gt_boxes[gm])
        if ap is not None:
            aps.append(ap)
    return float(np.mean(aps)) if aps else None


# ----------------------------------------------------------------------------
# io
# ----------------------------------------------------------------------------
def load_gt(label_path, w, h):
    """YOLO txt (cls cx cy w h, normalized) -> (boxes xyxy px, cls)"""
    if not os.path.isfile(label_path):
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int32)
    boxes, clss = [], []
    with open(label_path) as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            c, cx, cy, bw, bh = int(float(p[0])), *map(float, p[1:5])
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h])
            clss.append(c)
    if not boxes:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int32)
    return np.array(boxes, np.float32), np.array(clss, np.int32)


def collect_images(d):
    return sorted(p for p in Path(d).iterdir() if p.suffix.lower() in IMG_EXTS)


def label_path_for(img_path, labels_dir):
    return str(Path(labels_dir) / (Path(img_path).stem + '.txt'))


def run_model(weights, image_paths):
    """-> (dict[filename, (boxes, scores, cls)], names)"""
    model = YOLO(weights)
    out = {}

    # a .txt manifest keeps LoadImagesAndVideos lazy; a list source does not
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
        f.write('\n'.join(str(p) for p in image_paths))
        list_file = f.name

    try:
        stream = model.predict(
            source=list_file,
            imgsz=IMGSZ, conf=EVAL_CONF, iou=NMS_IOU, max_det=MAX_DET,
            device=DEVICE, stream=True, verbose=False,
        )
        for i, r in enumerate(stream, 1):
            b = r.boxes
            out[Path(r.path).name] = (
                b.xyxy.cpu().numpy().astype(np.float32),
                b.conf.cpu().numpy().astype(np.float32),
                b.cls.cpu().numpy().astype(np.int32),
            )
            if i % 250 == 0:
                print(f'  {i}/{len(image_paths)}')
    finally:
        os.unlink(list_file)

    names = dict(model.names)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out, names


def run_all(image_paths):
    """Runs each unique checkpoint once, even if several keys share a path."""
    by_path = {}
    for tag, w in WEIGHTS.items():
        by_path.setdefault(os.path.realpath(w), []).append(tag)
    dupes = {p: t for p, t in by_path.items() if len(t) > 1}
    for p, tags in dupes.items():
        print(f'WARNING: {tags} share the same checkpoint -> {p}')

    cache, preds, names = {}, {}, None
    for tag, w in WEIGHTS.items():
        rp = os.path.realpath(w)
        if rp in cache:
            print(f'{tag}: reusing inference from an identical checkpoint')
            preds[tag] = cache[rp]
            continue
        print(f'running {tag} ...')
        p, n = run_model(w, image_paths)
        cache[rp] = preds[tag] = p
        if names is None:
            names = n
        elif n != names:
            print(f'WARNING: {tag} has different class names than the first model')
    return preds, names


# ----------------------------------------------------------------------------
# drawing
# ----------------------------------------------------------------------------
FONT = cv2.FONT_HERSHEY_SIMPLEX


def scale_for(width):
    """Text/line scale so annotations stay readable at any resolution."""
    return max(1.0, width / 900.0) * FONT_SCALE


def put_label(img, text, x, y, color, s, above=True):
    """Filled label box anchored at (x, y). above=True -> box sits on top of y."""
    fs = 0.65 * s
    th = max(2, int(1.6 * s))
    (tw, tht), base = cv2.getTextSize(text, FONT, fs, th)
    pad = int(4 * s)
    if above:
        y1, y2 = max(0, y - tht - base - 2 * pad), max(tht + base, y)
    else:
        y1, y2 = y, y + tht + base + 2 * pad
    y2 = min(y2, img.shape[0] - 1)
    x2 = min(x + tw + 2 * pad, img.shape[1] - 1)
    cv2.rectangle(img, (x, y1), (x2, y2), color, -1)
    cv2.putText(img, text, (x + pad, y2 - base - pad), FONT, fs,
                (255, 255, 255), th, cv2.LINE_AA)


def draw_panel(img, pred, gt, names, title, ap):
    boxes, scores, cls = pred
    gt_boxes, gt_cls = gt

    keep = scores >= DRAW_CONF
    boxes, scores, cls = boxes[keep], scores[keep], cls[keep]

    # only classes that this image's GT actually contains can move its AP50
    if DRAW_ONLY_GT_CLASSES:
        keep = np.isin(cls, np.unique(gt_cls))
        boxes, scores, cls = boxes[keep], scores[keep], cls[keep]

    # downscale to PANEL_MAX_W, boxes follow
    canvas = img.copy()
    if PANEL_MAX_W and canvas.shape[1] > PANEL_MAX_W:
        f = PANEL_MAX_W / canvas.shape[1]
        canvas = cv2.resize(canvas, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        boxes, gt_boxes = boxes * f, gt_boxes * f

    tp = np.zeros(len(boxes), bool)
    gt_hit = np.zeros(len(gt_boxes), bool)
    if len(cls) or len(gt_cls):
        for c in np.unique(np.concatenate([cls, gt_cls])):
            pm, gm = cls == c, gt_cls == c
            t, m = greedy_match(boxes[pm], scores[pm], gt_boxes[gm])
            tp[np.where(pm)[0]] = t
            gt_hit[np.where(gm)[0]] = m

    s = scale_for(canvas.shape[1])
    lw = max(2, int(2 * s))

    for (x1, y1, x2, y2), c, hit in zip(gt_boxes.astype(int), gt_cls, gt_hit):
        color = C_GT_MATCHED if hit else C_GT_MISSED
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, lw if hit else lw + 1)
        if DRAW_GT_LABELS:
            put_label(canvas, f'GT {names.get(int(c), c)}', x1, y2, color, s, above=False)

    for (x1, y1, x2, y2), sc, c, is_tp in zip(boxes.astype(int), scores, cls, tp):
        color = C_TP if is_tp else C_FP
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, lw)
        put_label(canvas, f'{names.get(int(c), c)} {sc:.2f}', x1, y1, color, s, above=True)

    n_tp, n_fp, n_fn = int(tp.sum()), int((~tp).sum()), int((~gt_hit).sum())
    hdr_h = int(80 * s)
    header = np.full((hdr_h, canvas.shape[1], 3), C_BG, np.uint8)
    cv2.putText(header, f'{title}    AP50 = {ap:.3f}', (int(10 * s), int(30 * s)),
                FONT, 0.85 * s, (255, 255, 255), max(2, int(2.0 * s)), cv2.LINE_AA)
    cv2.putText(header, f'TP {n_tp}    FP {n_fp}    FN {n_fn}    (conf >= {DRAW_CONF})',
                (int(10 * s), int(60 * s)),
                FONT, 0.65 * s, (195, 195, 195), max(2, int(2.0 * s)), cv2.LINE_AA)
    return np.vstack([header, canvas])


def legend_bar(width, panel_w):
    s = scale_for(panel_w)
    h = int(40 * s)
    bar = np.full((h, width, 3), C_BG, np.uint8)
    items = [('GT matched (found)', C_GT_MATCHED),
             ('GT missed = FN', C_GT_MISSED),
             ('pred TP (hits a GT)', C_TP),
             ('pred FP (hits nothing)', C_FP)]
    fs, th = 0.55 * s, max(2, int(2.0 * s))
    x, sq = int(12 * s), int(20 * s)
    ytop = (h - sq) // 2
    for text, color in items:
        cv2.rectangle(bar, (x, ytop), (x + sq, ytop + sq), color, -1)
        x += sq + int(8 * s)
        cv2.putText(bar, text, (x, ytop + sq - int(3 * s)), FONT, fs,
                    (220, 220, 220), th, cv2.LINE_AA)
        x += cv2.getTextSize(text, FONT, fs, th)[0][0] + int(28 * s)
    return bar


def make_grid(panels):
    ph, pw = panels[0].shape[:2]
    blank = np.full((ph, pw, 3), C_BG, np.uint8)
    padded = panels + [blank] * (-len(panels) % GRID_COLS)

    vsep = np.full((ph, 8, 3), C_BG, np.uint8)
    rows = []
    for i in range(0, len(padded), GRID_COLS):
        chunk = padded[i:i + GRID_COLS]
        row = chunk[0]
        for p in chunk[1:]:
            row = np.hstack([row, vsep, p])
        rows.append(row)

    hsep = np.full((8, rows[0].shape[1], 3), C_BG, np.uint8)
    grid = rows[0]
    for r in rows[1:]:
        grid = np.vstack([grid, hsep, r])
    return np.vstack([grid, legend_bar(grid.shape[1], pw)])


# ----------------------------------------------------------------------------
def main():
    tags = list(WEIGHTS)
    assert REFERENCE in WEIGHTS, f'REFERENCE {REFERENCE!r} not in WEIGHTS'
    assert RANK_BY == 'spread' or RANK_BY in WEIGHTS, f'bad RANK_BY: {RANK_BY!r}'

    labels_dir = VAL_LABELS_DIR or VAL_IMAGES_DIR.replace('/images', '/labels')
    for k, v in WEIGHTS.items():
        assert os.path.isfile(v), f'{k}: not a file -> {v}\n  (needs to end in .pt)'
    assert os.path.isdir(VAL_IMAGES_DIR), VAL_IMAGES_DIR
    assert os.path.isdir(labels_dir), labels_dir

    out_img_dir = Path(OUTPUT_DIR) / 'top_diff'
    out_img_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(VAL_IMAGES_DIR)
    print(f'{len(images)} validation images, {len(tags)} models')

    preds, names = run_all(images)

    rows = []
    for p in images:
        key = p.name
        if any(key not in preds[t] for t in tags):
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt_boxes, gt_cls = load_gt(label_path_for(p, labels_dir), w, h)
        if SKIP_EMPTY_GT and len(gt_boxes) == 0:
            continue

        ap = {}
        for t in tags:
            b, s, c = preds[t][key]
            ap[t] = image_ap50(b, s, c, gt_boxes, gt_cls)
        if any(v is None for v in ap.values()):
            continue

        vals = list(ap.values())
        rank_value = (max(vals) - min(vals) if RANK_BY == 'spread'
                      else ap[RANK_BY] - ap[REFERENCE])

        row = {'image': key, 'n_gt': len(gt_boxes)}
        row.update({f'ap50_{t}': ap[t] for t in tags})
        row.update({f'vs_ref_{t}': ap[t] - ap[REFERENCE] for t in tags if t != REFERENCE})
        row['rank_value'] = rank_value
        rows.append(row)

    if not rows:
        raise SystemExit('no images with ground truth were scored -- check labels_dir')

    rows.sort(key=lambda r: {'abs': -abs(r['rank_value']),
                             'a_wins': -r['rank_value'],
                             'b_wins': r['rank_value']}[SORT_BY])

    csv_path = Path(OUTPUT_DIR) / 'per_image_ap50.csv'
    with open(csv_path, 'w', newline='') as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)
    print(f'wrote {csv_path}')

    for rank, r in enumerate(rows[:TOP_K], 1):
        p = Path(VAL_IMAGES_DIR) / r['image']
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        gt = load_gt(label_path_for(p, labels_dir), w, h)

        panels = [draw_panel(img, preds[t][p.name], gt, names, t, r[f'ap50_{t}'])
                  for t in tags]
        name = f"{rank:02d}_{RANK_BY}{r['rank_value']:+.3f}_{p.stem}.jpg"
        cv2.imwrite(str(out_img_dir / name), make_grid(panels))

    print(f'wrote {min(TOP_K, len(rows))} images -> {out_img_dir}')
    print()
    for t in tags:
        print(f'mean AP50  {t:<32} {np.mean([r[f"ap50_{t}"] for r in rows]):.4f}')


if __name__ == '__main__':
    main()