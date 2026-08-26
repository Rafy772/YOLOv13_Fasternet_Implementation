from ultralytics import YOLO
import csv
import torch

# ============================== CONFIG ==============================
# label -> checkpoint path  (add as many as you want to compare)
WEIGHTS = {
    'yolov13_fasternet': '/home/user129/yolov13/Paper_Hasil_Ujicoba/Dolphin14k_Hasil Percobaan/yolov13_Partial Convolution_640/train186/weights/best.pt',
    'yolov13_baseline': '/home/user129/yolov13/Paper_Hasil_Ujicoba/Dolphin14k_Hasil Percobaan/yolov13n_baseline_640/weights/best.pt',
}

DATA      = 'ultralytics/cfg/datasets/lumba_lumba.yaml'   # TT100K.yaml, lumba_lumba.yaml, visdrone2019.yaml
IMGSZ     = 640
BATCH     = 30
DEVICE    = '0'          # '0' = GPU 0  |  'cpu' = CPU   (NOTE: '0' is GPU, not CPU)
PLOTS     = True
SAVE_JSON = True
VERBOSE   = True

PROJECT   = 'val_multi'         # per-model val outputs (plots/json) go under this folder
OUT_CSV   = 'val_results.csv'
ROUND     = 4
# ====================================================================

FIELDS = ['model', 'weights', 'precision', 'recall', 'f1',
          'mAP50', 'mAP50-95', 'inference_ms']

with open(OUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    f.flush()

    for name, ckpt in WEIGHTS.items():
        print(f'\n===== Validating: {name} =====')
        model = None
        try:
            model = YOLO(ckpt)
            metrics = model.val(
                data=DATA,
                imgsz=IMGSZ,
                batch=BATCH,
                plots=PLOTS,
                save_json=SAVE_JSON,
                verbose=VERBOSE,
                device=DEVICE,
                project=PROJECT,
                name=name,
                exist_ok=True,     # overwrite this model's folder on re-runs
            )

            p   = metrics.box.mp        # mean precision
            r   = metrics.box.mr        # mean recall
            f1  = 2 * p * r / (p + r + 1e-16)
            m50 = metrics.box.map50     # mAP@0.50
            m95 = metrics.box.map       # mAP@0.50-0.95

            speed  = getattr(metrics, 'speed', {}) or {}
            inf_ms = speed.get('inference', float('nan'))

            print(f'P: {p:.4f} | R: {r:.4f} | F1: {f1:.4f} | '
                  f'mAP50: {m50:.4f} | mAP50-95: {m95:.4f} | inf: {inf_ms:.2f} ms/img')

            row = {
                'model': name, 'weights': ckpt,
                'precision': round(p, ROUND), 'recall': round(r, ROUND),
                'f1': round(f1, ROUND), 'mAP50': round(m50, ROUND),
                'mAP50-95': round(m95, ROUND), 'inference_ms': round(inf_ms, 2),
            }
        except Exception as e:
            print(f'[FAILED] {name}: {e}')
            row = {'model': name, 'weights': ckpt, 'precision': '', 'recall': '',
                   'f1': '', 'mAP50': '', 'mAP50-95': '', 'inference_ms': ''}
        finally:
            if model is not None:
                del model
            torch.cuda.empty_cache()

        writer.writerow(row)
        f.flush()   # persist after each model -> survives a WSL OOM crash

print(f'\nSaved -> {OUT_CSV}')