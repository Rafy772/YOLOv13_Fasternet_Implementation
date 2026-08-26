from ultralytics import YOLO

model = YOLO('yolov13.yaml') #yolov13-fasterall-p2.yaml, yolov13-fasterall.yaml, yolov13-Semua Kombinasi.yaml

# Train the model
results = model.train(
    data='ultralytics/cfg/datasets/lumba_lumba.yaml', #TT100K.yaml, UAVDT.yaml, visdrone2019.yaml, lumba_lumba.yaml
    epochs=1,
    batch=24,
    imgsz=640,
    scale=0.5,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.1,
    device=0,              # ✅ single GPU
    patience=120, 
    pretrained=False,
)
