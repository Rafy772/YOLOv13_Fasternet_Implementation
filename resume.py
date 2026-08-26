from ultralytics import YOLO

model = YOLO('/home/rafy/yolov13/runs/detect/train298/weights/last.pt')

results = model.train(resume=True)