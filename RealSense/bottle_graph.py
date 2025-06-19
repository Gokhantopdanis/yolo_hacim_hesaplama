import cv2
import cvzone
from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
from constants import *

# RealSense pipeline başlatma
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config)

# Hizalayıcı oluşturma (derinlik ve renk görüntülerini hizalamak için)
align_to = rs.stream.color
align = rs.align(align_to)

model = YOLO("yolov8l-seg.pt")

depth_values = []
widths = []
heights = []

try:
    while True:
        # Kameradan kare alma
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue
        
        # Derinlik ve renk görüntülerini numpy dizilerine dönüştürme
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Kameranın intrinsics değerlerini alma
        color_intrinsics = aligned_frames.get_color_frame().get_profile().as_video_stream_profile().get_intrinsics()
        focal_length = color_intrinsics.fx

        # YOLO modelini kullanarak şişeyi tespit etme
        results = model(color_image, stream=True, conf=0.3, save=False, show=True, save_txt=False, verbose=False, classes=39)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                cls_name = f"{classNames[cls]}"

                if cls_name == "bottle":
                    # Bounding Box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    cvzone.cornerRect(color_image, (x1, y1, w, h), colorR=(255, 255, 0), colorC=(255, 0, 0))

                    # Şişenin orta noktasını hesaplama
                    x_mid = int((x1 + x2) / 2)
                    y_mid = int((y1 + y2) / 2)

                    # Derinlik bilgisini alma
                    depth_value = depth_frame.get_distance(x_mid, y_mid) * 100

                    # Gerçek genişlik ve yükseklik hesaplama
                    width_cm = ((w * depth_value) / focal_length)
                    height_cm = ((h * depth_value) / focal_length)

                    # Değerleri listeye ekleme
                    depth_values.append(depth_value)
                    widths.append(width_cm)
                    heights.append(height_cm)

                    # Bilgileri Görüntüye Yazdırma
                    cvzone.putTextRect(color_image, f'W: {width_cm:.1f}cm, H: {height_cm:.1f}cm',
                                   (x1, y1 - 40), scale=1.5, colorR=(0, 0, 0))

                    cv2.circle(color_image, (x_mid, y_mid), 5, (0, 255, 0), 2)
                    cvzone.putTextRect(color_image, f'Depth: {depth_value:.2f}cm', (x_mid, y_mid - 10), scale=1.5, colorR=(0, 255, 0))

        # Görüntüyü göster
        cv2.imshow("Image Detector", color_image)
        if cv2.waitKey(30) == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

# Depth ve boyut değerlerini grafik olarak çizme
plt.figure(figsize=(12, 6))

# Şişe genişliğinin derinliğe bağlı grafiği
plt.subplot(1, 2, 1)
plt.plot(depth_values, widths, label='Width (cm)', marker='o', color='blue')
plt.xlabel('Depth (cm)')
plt.ylabel('Width (cm)')
plt.title('Bottle Width vs Depth')
plt.grid(True)
plt.legend()

# Şişe yüksekliğinin derinliğe bağlı grafiği
plt.subplot(1, 2, 2)
plt.plot(depth_values, heights, label='Height (cm)', marker='o', color='green')
plt.xlabel('Depth (cm)')
plt.ylabel('Height (cm)')
plt.title('Bottle Height vs Depth')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()