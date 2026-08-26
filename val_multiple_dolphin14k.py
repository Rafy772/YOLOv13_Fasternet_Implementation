"""
Validate multiple YOLO checkpoints on the same dataset and export metrics to CSV.
"""

import csv
import gc
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, get_num_params

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
ROOT = Path('/home/rafy/yolov13/Paper_Hasil_Ujicoba/Dolphin14k_Hasil Percobaan')

WEIGHTS = {
    'yolov13_baseline':            ROOT / 'yolov13n_baseline_640/weights/best.pt',
    'yolov13_Stride 1':            ROOT / 'yolov13n_Stride 1_640/train95/weights/best.pt',
    'yolov13_Kernel 5':            ROOT / 'yolov13n_Kernel 5_640/train104/weights/best.pt',
    'yolov13_P2':                  ROOT / 'yolov13n_P2_640/train98/weights/best.pt',
    'yolov13_Partial Convolution': ROOT / 'yolov13_Partial Convolution_640/train186/weights/best.pt',
    'yolov13_Semua Perubahan':     ROOT / 'yolov13n_Semua Kombinasi_Partial Convolution juga_640/train230/weights/best.pt',
}

DATA        = 'ultralytics/cfg/datasets/lumba_lumba.yaml'  # TT100K.yaml, lumba_lumba.yaml, visdrone2019.yaml
IMGSZ       = 640
BATCH       = 1
DEVICE      = '0'          # '0' = GPU 0, 'cpu' = CPU
PLOTS       = True
SAVE_JSON   = True
VERBOSE     = True

PROJECT     = 'runs/val_compare'   # each model gets its own subfolder here
OUT_CSV     = 'val_summary.csv'
OUT_CSV_CLS = 'val_per_class.csv'

EPS = 1e-16

# ----------------------------------------------------------------------------
# VALIDATION LOOP
# ----------------------------------------------------------------------------
summary_rows = []
per_class_rows = []

for name, wpath in WEIGHTS.items():
    wpath = Path(wpath)
    print(f'\n{"=" * 78}\n[{name}] {wpath}\n{"=" * 78}')

    if not wpath.is_file():
        print(f'  !! SKIPPED: not a file -> {wpath}')
        continue

    model = YOLO(str(wpath))

    # complexity: measure BEFORE val() — val() fuses Conv+BN in-place
    n_params = get_num_params(model.model)
    try:
        gflops = get_flops(model.model, imgsz=IMGSZ)
    except Exception as e:
        print(f'  !! GFLOPs unavailable ({e}) -- pip install thop')
        gflops = float('nan')

    metrics = model.val(
        data=DATA,
        imgsz=IMGSZ,
        batch=BATCH,
        plots=PLOTS,
        save_json=SAVE_JSON,
        verbose=VERBOSE,
        device=DEVICE,
        project=PROJECT,
        name=name.replace(' ', '_'),   # keeps plots/json separate per model
        exist_ok=True,
    )

    p = float(metrics.box.mp)
    r = float(metrics.box.mr)
    f1 = 2 * p * r / (p + r + EPS)

    speed = getattr(metrics, 'speed', {}) or {}
    pre_ms = speed.get('preprocess', 0.0)
    inf_ms = speed.get('inference', 0.0)
    post_ms = speed.get('postprocess', 0.0)
    total_ms = pre_ms + inf_ms + post_ms

    summary_rows.append({
        'model': name,
        'weights': str(wpath),
        'params_M': round(n_params / 1e6, 3),
        'GFLOPs': round(gflops, 2),
        'precision': round(p, 4),
        'recall': round(r, 4),
        'f1': round(f1, 4),
        'mAP50': round(float(metrics.box.map50), 4),
        'mAP50-95': round(float(metrics.box.map), 4),
        'mAP75': round(float(metrics.box.map75), 4),
        'fitness': round(float(metrics.fitness), 4),
        'preprocess_ms': round(pre_ms, 3),
        'inference_ms': round(inf_ms, 3),
        'postprocess_ms': round(post_ms, 3),
        'total_ms': round(total_ms, 3),
        'fps_inference': round(1000.0 / inf_ms, 2) if inf_ms > 0 else 0.0,
        'fps_end2end': round(1000.0 / total_ms, 2) if total_ms > 0 else 0.0,
    })

    # per-class breakdown: metrics.box.ap_class_index maps rows -> class ids
    names = metrics.names if hasattr(metrics, 'names') else model.names
    for i, c in enumerate(metrics.box.ap_class_index):
        cp, cr, cap50, cap = (float(x[i]) for x in
                              (metrics.box.p, metrics.box.r, metrics.box.ap50, metrics.box.ap))
        per_class_rows.append({
            'model': name,
            'class_id': int(c),
            'class_name': names[int(c)],
            'precision': round(cp, 4),
            'recall': round(cr, 4),
            'f1': round(2 * cp * cr / (cp + cr + EPS), 4),
            'AP50': round(cap50, 4),
            'AP50-95': round(cap, 4),
        })

    print(f'  Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f} '
          f'| mAP50: {metrics.box.map50:.4f} | mAP50-95: {metrics.box.map:.4f}')

    # free VRAM before the next checkpoint
    del model, metrics
    gc.collect()
    torch.cuda.empty_cache()

# ----------------------------------------------------------------------------
# EXPORT
# ----------------------------------------------------------------------------
if summary_rows:
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f'\nWrote {OUT_CSV} ({len(summary_rows)} models)')

if per_class_rows:
    with open(OUT_CSV_CLS, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(per_class_rows[0].keys()))
        w.writeheader()
        w.writerows(per_class_rows)
    print(f'Wrote {OUT_CSV_CLS} ({len(per_class_rows)} rows)')

# console tables
if summary_rows:
    print()
    hdr = (f'{"model":<32} {"P":>7} {"R":>7} {"F1":>7} {"mAP50":>7} {"mAP50-95":>9}')
    print(hdr)
    print('-' * len(hdr))
    for row in summary_rows:
        print(f'{row["model"]:<32} {row["precision"]:>7.4f} {row["recall"]:>7.4f} '
              f'{row["f1"]:>7.4f} {row["mAP50"]:>7.4f} {row["mAP50-95"]:>9.4f}')

    print()
    hdr2 = (f'{"model":<32} {"Params(M)":>10} {"GFLOPs":>8} {"Infer(ms)":>10} '
            f'{"Total(ms)":>10} {"FPS":>8} {"FPS(e2e)":>9}')
    print(hdr2)
    print('-' * len(hdr2))
    for row in summary_rows:
        print(f'{row["model"]:<32} {row["params_M"]:>10.3f} {row["GFLOPs"]:>8.2f} '
              f'{row["inference_ms"]:>10.2f} {row["total_ms"]:>10.2f} '
              f'{row["fps_inference"]:>8.2f} {row["fps_end2end"]:>9.2f}')

    print(f'\n(latency measured at imgsz={IMGSZ}, batch={BATCH}, device={DEVICE})')