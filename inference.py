from ultralytics import YOLO # Import kelas YOLO dari library Ultralytics

# Load your trained model
model = YOLO(
    "/home/rafy/yolov13/Paper_Hasil_Ujicoba/Visdrone2019_Hasil Percobaan/Partial Convolution_yolov13n_640/train212/weights/best.pt"
)

# Run inference on a video and save the result
model.predict(
    source="/home/rafy/yolov13/visdrone2019.mp4",  # Path video input
    save=True,        # Simpan video ber-anotasi ke folder runs/detect/predict/
    show=True,        # Tampilkan video secara real-time selama diproses (butuh GUI/display)
    conf=0.25,        # Ambang confidence minimum; deteksi di bawah 0.25 dibuang
    imgsz=640,        # Resize frame ke 640px — samakan dengan ukuran saat training
    device=0,         # Pakai GPU index 0 (CUDA); ganti "cpu" kalau tanpa GPU
    half=True,        # Pakai FP16 (half precision) — lebih cepat & hemat VRAM di GPU
    verbose=False     # Matikan log detail per frame (output lebih bersih)
)

print("Done!") # Penanda proses selesai