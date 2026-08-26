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

Grad-CAM notes:
  - Target layer = the feature pyramid fed into the Detect head (P3/P4/P5,
    plus P2 for the P2 variant), captured via a forward-pre-hook. No
    hard-coded layer indices, so it works across all variants.
  - Target scalar = sum of class probabilities of confident anchors
    (>= CAM_CONF). CAM_TARGET_CLASS=None is class-agnostic; set an int
    to explain a single class.
  - Needs its own gradient forward pass per (image, model) — the no-grad
    predict() results cannot be reused. Use CAM_TOP_K to limit cost.
  - Heatmaps are coarse at VisDrone's scale (feature maps are 80x80 at
    finest). Read them qualitatively for attention region comparison.
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
ROOT = Path('/home/rafy/yolov13/Paper_Hasil_Ujicoba/TT100K_Hasil Percobaan')

WEIGHTS = {
    'yolov13_baseline': ROOT / 'yolov13n_640_Baseline/weights/best.pt',
    'yolov13_Stride 1': ROOT / 'yolov13n_640_Stride 1/train91/weights/best.pt',
    'yolov13_Kernel 5': ROOT / 'yolov13n_640_Kernel 5/train103/weights/best.pt',
    'yolov13_P2': ROOT / 'yolov13n_640_P2/train99/weights/best.pt',
    'yolov13_Partial Convolution': ROOT / 'yolov13n_640_Partial Convolution/train193/weights/best.pt',
    'yolov13_S1 + K5 + P2': ROOT / 'yolov13n_640_Semua Kombinasi_Partial Convolution/train235/weights/best.pt',
    'yolov13_Semua Perubahan': ROOT / 'yolov13n_640_Semua Kombinasi/weights/best.pt',
}

REFERENCE = 'yolov13_baseline'   # the model everything is compared against
RANK_BY = 'spread'               # 'spread' (max-min across all models)
                                 #   or a model key -> that model minus REFERENCE
SORT_BY = 'abs'                  # 'abs' | 'a_wins' (rank value high) | 'b_wins' (low)

VAL_IMAGES_DIR = '/home/rafy/yolov13/TT100K/val/images'
VAL_LABELS_DIR = '/home/rafy/yolov13/TT100K/val/labels'
OUTPUT_DIR = '/home/rafy/yolov13/ap50_diff_analysis'

IMGSZ = 640
DEVICE = 0
DRAW_CONF = 0.25                 # detections drawn on the output images
EVAL_CONF = DRAW_CONF            # set to 0.001 for standard mAP protocol
NMS_IOU = 0.7
MAX_DET = 300
MATCH_IOU = 0.5

TOP_K = 5
SKIP_EMPTY_GT = True             # images with no GT have undefined AP -> skip

DRAW_ONLY_GT_CLASSES = True      # hide predictions of classes absent from GT
DRAW_GT_LABELS = True            # class name under each GT box
GRID_COLS = 3                    # panels per row (applies to both outputs)
PANEL_MAX_W = 1100               # downscale each panel to this width
FONT_SCALE = 1.0                 # bump if text is still too small

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

# BGR colours for the detection grids
C_GT_MATCHED = (120, 220, 120)
C_GT_MISSED  = (0, 215, 255)
C_TP         = (255, 160, 0)
C_FP         = (60, 60, 255)
C_BG         = (32, 32, 32)

# --- Grad-CAM ---
DO_GRADCAM       = True
CAM_TOP_K        = 500            # one grad fwd+bwd per (image, model) -- keep small
CAM_CONF         = DRAW_CONF     # anchors with cls prob >= this define the target
CAM_TARGET_CLASS = None          # None = class-agnostic; or int class id
CAM_LEVEL        = 'finest'      # 'finest' | 'max' | 'mean' over head-input levels
CAM_ALPHA        = 0.5           # heatmap opacity blended over original image
CAM_DRAW_GT      = True          # draw thin white GT boxes over the heatmap
CAM_DRAW_PRED    = False         # draw thin black predicted boxes over the heatmap
CAM_COLORMAP     = cv2.COLORMAP_JET


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
# drawing helpers
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

    if DRAW_ONLY_GT_CLASSES:
        keep = np.isin(cls, np.unique(gt_cls))
        boxes, scores, cls = boxes[keep], scores[keep], cls[keep]

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
# Grad-CAM
# Target layer  = feature pyramid fed into the Detect head (P3/P4/P5, +P2 for
#                 P2 variant), grabbed with a forward-pre-hook — no hard-coded
#                 indices, so it works across all seven variants unchanged.
# Target scalar = sum of class probabilities of confident anchors (>= CAM_CONF).
#                 CAM_TARGET_CLASS=None is class-agnostic; set an int to explain
#                 one class only.
# ----------------------------------------------------------------------------
def _letterbox(im, new_shape=IMGSZ, color=(114, 114, 114)):
    """Replicate ultralytics letterbox -> (padded_img, ratio, (pad_left, pad_top))."""
    shape = im.shape[:2]  # h, w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (round(shape[1] * r), round(shape[0] * r))  # w, h
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if (shape[1], shape[0]) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)


def _find_detect_head(nn_model):
    """Locate the Detect head — the module the pyramid is passed into."""
    seq = getattr(nn_model, 'model', None)
    cand = seq[-1] if hasattr(seq, '__getitem__') else None
    if cand is not None and (hasattr(cand, 'nl')
                             or cand.__class__.__name__.lower().startswith('detect')):
        return cand
    # fallback: scan all modules for last Detect-like
    for m in nn_model.modules():
        if hasattr(m, 'nl') or m.__class__.__name__.lower().startswith('detect'):
            cand = m
    if cand is None:
        raise RuntimeError('Grad-CAM: could not locate a Detect head — '
                           'check your YOLOv13 fork\'s head class name')
    return cand


class GradCAM:
    """Grad-CAM over the detection head's input pyramid.

    Create once per model (registers a hook), call on BGR images, then
    .remove() to clean up. Requires one forward + backward per image.

    Quick smoke test before a full run:
        g = GradCAM(YOLO(str(WEIGHTS['yolov13_baseline'])))
        cam = g(cv2.imread('path/to/any_val_image.jpg'))
        print('levels:', len(g._acts),
              'grads:', [f.grad is not None for f in g._acts],
              'cam:', cam.shape, float(cam.min()), float(cam.max()))
        g.remove()
    Expect: levels=3 (4 for the P2 model), all grads True, cam spanning ~0->1.
    If grads come back None, the head output format differs from stock
    ultralytics — open an issue or adjust _build_target() below.
    """

    def __init__(self, yolo,
                 device=DEVICE,
                 target_class=CAM_TARGET_CLASS,
                 conf=CAM_CONF,
                 level=CAM_LEVEL,
                 imgsz=IMGSZ):
        self.nn    = yolo.model
        self.names = yolo.names
        self.device = f'cuda:{device}' if isinstance(device, int) else device
        self.nn.to(self.device).float().eval()
        self.nc           = len(yolo.names)
        self.target_class = target_class
        self.conf         = conf
        self.level        = level
        self.imgsz        = imgsz
        self._acts        = []
        self._handle      = _find_detect_head(self.nn).register_forward_pre_hook(
            self._pre_hook)

    # ------------------------------------------------------------------
    def _pre_hook(self, module, args):
        # args[0] is the list [P3, P4, P5(, P2)] Detect.forward receives.
        # The head overwrites x in place (x[i] = cat(...)), so capture the
        # tensors now. retain_grad() lets us read .grad after backward.
        self._acts = [f for f in args[0]]
        for f in self._acts:
            if f.requires_grad:
                f.retain_grad()

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    # ------------------------------------------------------------------
    @staticmethod
    def _norm(arr):
        rng = arr.max() - arr.min()
        return (arr - arr.min()) / rng if rng > 0 else np.zeros_like(arr)

    def _build_target(self, y):
        """Sum of confident class probabilities -> scalar for backward."""
        # y expected shape: (1, 4+nc, N)  (anchors in last dim)
        if y.dim() != 3:
            raise RuntimeError(
                f'Grad-CAM: head output shape {tuple(y.shape)} is unexpected. '
                'YOLOv13 may return (preds, feats) — if so, unpack out[0].')
        if y.shape[1] != self.nc + 4 and y.shape[2] == self.nc + 4:
            y = y.transpose(1, 2)                 # -> (1, 4+nc, N)
        cls = y[0, 4:4 + self.nc, :]              # sigmoid class probs (nc, N)
        scores = (cls.max(0).values if self.target_class is None
                  else cls[self.target_class])     # (N,)
        mask = scores >= self.conf
        return (scores[mask].sum() if int(mask.sum())
                else scores.topk(min(10, scores.numel())).values.sum())

    # ------------------------------------------------------------------
    def __call__(self, bgr):
        """bgr: H×W×3 uint8  ->  cam: H×W float32 in [0, 1]"""
        lb, r, (padx, pady) = _letterbox(bgr, self.imgsz)
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        x = (torch.from_numpy(rgb).permute(2, 0, 1)[None]
             .float().to(self.device) / 255.0)
        x.requires_grad_(True)

        self.nn.zero_grad(set_to_none=True)
        self._acts = []
        with torch.enable_grad():
            out = self.nn(x)

        # Handle (preds, feats) tuple returned by some ultralytics heads
        y = out[0] if isinstance(out, (list, tuple)) else out
        self._build_target(y).backward()

        H, W = lb.shape[:2]
        levels = []
        for f in self._acts:
            if f.grad is None:
                continue
            # Grad-CAM: weight each channel by its gradient's global avg pooling
            w_c  = f.grad.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
            cam  = torch.relu((w_c * f).sum(1))[0]          # (h, w)
            levels.append(cam.detach().float().cpu().numpy())

        if not levels:
            return np.zeros(bgr.shape[:2], np.float32)

        # Aggregate levels -> single (H_lb, W_lb) map
        if self.level == 'finest':
            cam_lb = cv2.resize(
                self._norm(max(levels, key=lambda c: c.size)),
                (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            ups = [cv2.resize(self._norm(c), (W, H), interpolation=cv2.INTER_LINEAR)
                   for c in levels]
            cam_lb = np.max(ups, 0) if self.level == 'max' else np.mean(ups, 0)

        # Undo letterbox padding -> original image space
        h0, w0 = bgr.shape[:2]
        uw, uh = round(w0 * r), round(h0 * r)
        crop   = cam_lb[max(0, pady):min(H, pady + uh),
                        max(0, padx):min(W, padx + uw)]
        cam0   = cv2.resize(crop, (w0, h0), interpolation=cv2.INTER_LINEAR)
        return self._norm(cam0).astype(np.float32)


# ----------------------------------------------------------------------------
# Grad-CAM drawing
# ----------------------------------------------------------------------------
def overlay_cam(bgr, cam, alpha=CAM_ALPHA):
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), CAM_COLORMAP)
    return cv2.addWeighted(heat, alpha, bgr, 1 - alpha, 0)


def draw_cam_panel(img, cam, gt, names, title, ap, pred=None):
    """One Grad-CAM panel: heatmap overlay + optional GT/pred boxes + header."""
    gt_boxes, gt_cls = gt
    canvas = overlay_cam(img, cam)

    f = 1.0
    if PANEL_MAX_W and canvas.shape[1] > PANEL_MAX_W:
        f = PANEL_MAX_W / canvas.shape[1]
        canvas = cv2.resize(canvas, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)

    s  = scale_for(canvas.shape[1])
    lw = max(1, int(1.5 * s))

    # thin black predicted boxes (only confident, only GT classes)
    if CAM_DRAW_PRED and pred is not None:
        pb, psc, pc = pred
        keep = psc >= DRAW_CONF
        if DRAW_ONLY_GT_CLASSES:
            keep &= np.isin(pc, np.unique(gt_cls))
        for (x1, y1, x2, y2) in (pb[keep] * f).astype(int):
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 0), lw)

    # thin white GT boxes
    if CAM_DRAW_GT:
        for (x1, y1, x2, y2) in (gt_boxes * f).astype(int):
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), lw)

    hdr_h  = int(80 * s)
    header = np.full((hdr_h, canvas.shape[1], 3), C_BG, np.uint8)
    cv2.putText(header, f'{title}    AP50 = {ap:.3f}',
                (int(10 * s), int(30 * s)),
                FONT, 0.85 * s, (255, 255, 255), max(2, int(2.0 * s)), cv2.LINE_AA)
    tgt_str = 'all classes' if CAM_TARGET_CLASS is None else f'class {CAM_TARGET_CLASS}'
    cv2.putText(header,
                f'Grad-CAM  target={tgt_str}  level={CAM_LEVEL}  conf>={CAM_CONF}',
                (int(10 * s), int(60 * s)),
                FONT, 0.6 * s, (195, 195, 195), max(2, int(2.0 * s)), cv2.LINE_AA)
    return np.vstack([header, canvas])


def cam_legend_bar(width, panel_w):
    """Minimal legend for the Grad-CAM grids."""
    s   = scale_for(panel_w)
    h   = int(40 * s)
    bar = np.full((h, width, 3), C_BG, np.uint8)
    items = [('■ high activation', (0, 0, 200)),
             ('■ low activation',  (200, 0, 0))]
    if CAM_DRAW_GT:
        items.append(('□ GT box', (255, 255, 255)))
    if CAM_DRAW_PRED:
        items.append(('□ pred box', (0, 0, 0)))
    fs, th = 0.55 * s, max(2, int(2.0 * s))
    x, sq  = int(12 * s), int(20 * s)
    ytop   = (h - sq) // 2
    for text, color in items:
        cv2.putText(bar, text, (x, ytop + sq - int(3 * s)),
                    FONT, fs, color, th, cv2.LINE_AA)
        x += cv2.getTextSize(text, FONT, fs, th)[0][0] + int(28 * s)
    return bar


def make_cam_grid(panels):
    """Same layout as make_grid() but uses cam_legend_bar."""
    ph, pw  = panels[0].shape[:2]
    blank   = np.full((ph, pw, 3), C_BG, np.uint8)
    padded  = panels + [blank] * (-len(panels) % GRID_COLS)

    vsep = np.full((ph, 8, 3), C_BG, np.uint8)
    rows = []
    for i in range(0, len(padded), GRID_COLS):
        chunk = padded[i:i + GRID_COLS]
        row   = chunk[0]
        for p in chunk[1:]:
            row = np.hstack([row, vsep, p])
        rows.append(row)

    hsep = np.full((8, rows[0].shape[1], 3), C_BG, np.uint8)
    grid = rows[0]
    for r in rows[1:]:
        grid = np.vstack([grid, hsep, r])
    return np.vstack([grid, cam_legend_bar(grid.shape[1], pw)])


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

    # --- detection grids (original output) ---
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

    # --- Grad-CAM grids ---
    if DO_GRADCAM:
        cam_dir = Path(OUTPUT_DIR) / 'gradcam'
        cam_dir.mkdir(parents=True, exist_ok=True)
        print(f'\nbuilding Grad-CAM for top {min(CAM_TOP_K, len(rows))} images ...')

        # Load all models up front (nano weights are small; no OOM risk here)
        cams = {}
        for t in tags:
            print(f'  loading {t} for Grad-CAM ...')
            cams[t] = GradCAM(YOLO(str(WEIGHTS[t])))

        try:
            for rank, r in enumerate(rows[:CAM_TOP_K], 1):
                p   = Path(VAL_IMAGES_DIR) / r['image']
                img = cv2.imread(str(p))
                gt  = load_gt(label_path_for(p, labels_dir),
                              img.shape[1], img.shape[0])

                panels = []
                for t in tags:
                    cam   = cams[t](img)
                    panel = draw_cam_panel(
                        img, cam, gt, names, t, r[f'ap50_{t}'],
                        pred=preds[t][p.name])
                    panels.append(panel)

                name = (f"{rank:02d}_{RANK_BY}{r['rank_value']:+.3f}"
                        f"_{p.stem}_cam.jpg")
                cv2.imwrite(str(cam_dir / name), make_cam_grid(panels))

                if rank % 5 == 0:
                    print(f'  {rank}/{min(CAM_TOP_K, len(rows))}')
        finally:
            for c in cams.values():
                c.remove()

        print(f'wrote {min(CAM_TOP_K, len(rows))} Grad-CAM grids -> {cam_dir}')

    print()
    for t in tags:
        print(f'mean AP50  {t:<32} {np.mean([r[f"ap50_{t}"] for r in rows]):.4f}')


if __name__ == '__main__':
    main()