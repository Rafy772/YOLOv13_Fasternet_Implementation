from ultralytics import YOLO
import torch
import cv2
import matplotlib.pyplot as plt

# =========================
# Load model
# =========================
model = YOLO(
    r"C:\Users\Pongo\Downloads\Program Demo Live\yolov13\Paper_Hasil_Ujicoba\Visdrone2019_Hasil Percobaan\Semua Kombinasi_yolov13n_640\weights\best.pt"
)

device = 0 if torch.cuda.is_available() else "cpu"
half = torch.cuda.is_available()

print(f"Using device: {device}")

# =========================
# Input image & label
# =========================
image_path = (
    r"C:\Users\Pongo\Downloads\Program Demo Live\yolov13"
    r"\0000001_03499_d_0000006_jpg.rf.634d48e1bdf0429850be2a4f48786754.jpg"
)

label_path = image_path.replace(".jpg", ".txt")

# =========================
# Class names
# =========================
class_names = {
    0: "Pedestrian",
    1: "People",
    2: "Bicycle",
    3: "Car",
    4: "Van",
    5: "Truck",
    6: "Tricycle",
    7: "Awning-Tricycle",
    8: "Bus",
    9: "Motor"
}

# =========================
# Read image
# =========================
original = cv2.imread(image_path)

if original is None:
    raise FileNotFoundError(image_path)

gt_image = original.copy()

h, w = original.shape[:2]

# =========================
# Draw Ground Truth
# =========================
GT_COLOR = (0, 255, 0)      # Green (BGR)
THICKNESS = 2
FONT_SCALE = 0.5

with open(label_path, "r") as f:
    for line in f:
        values = line.strip().split()

        cls = int(values[0])
        xc, yc, bw, bh = map(float, values[1:])

        # Convert YOLO normalized format
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        cv2.rectangle(
            gt_image,
            (x1, y1),
            (x2, y2),
            GT_COLOR,
            THICKNESS
        )

        label = class_names.get(cls, str(cls))

        cv2.putText(
            gt_image,
            label,
            (x1, max(y1 - 5, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            GT_COLOR,
            1,
            cv2.LINE_AA
        )

# =========================
# YOLO Prediction
# =========================
results = model.predict(
    source=original,
    conf=0.25,
    imgsz=640,
    device=device,
    half=half,
    verbose=False
)

prediction = results[0].plot(
    font_size=0.5,
    line_width=2
)

# =========================
# Convert to RGB
# =========================
original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
gt_rgb = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
prediction_rgb = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)

# =========================
# Display
# =========================
plt.figure("Original Image", figsize=(8,6))
plt.imshow(original_rgb)
plt.title("Original Image")
plt.axis("off")

plt.figure("Ground Truth", figsize=(8,6))
plt.imshow(gt_rgb)
plt.title("Ground Truth")
plt.axis("off")

plt.figure("YOLO Detection", figsize=(8,6))
plt.imshow(prediction_rgb)
plt.title("YOLO Detection")
plt.axis("off")

print("Close all windows to exit.")
plt.show()

print("Done!")