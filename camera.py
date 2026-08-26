from ultralytics import YOLO
import torch

# Load your trained model
model = YOLO(
    r"C:\Users\Pongo\Downloads\Program Demo Live\yolov13\Paper_Hasil_Ujicoba\Visdrone2019_Hasil Percobaan\Partial Convolution_yolov13n_640\train212\weights\best.pt"
)

# Automatically choose GPU if available
device = 0 if torch.cuda.is_available() else "cpu"
half = torch.cuda.is_available()

print(f"Using device: {device}")

# Real-time inference from webcam
results = model.predict(
    source=1,          # Default webcam
    stream=True,       # Return generator
    show=True,         # Show live window
    conf=0.25,
    imgsz=640,
    device=device,
    half=half,
    verbose=False,
    save=False
)

# Keep the program alive by consuming the stream
for _ in results:
    pass

print("Done!")