
import cv2
import cvzone
from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
from constants import *
import torch
print(torch.cuda.is_available())
# RealSense pipeline başlatma
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# Hizalayıcı oluşturma (derinlik ve renk görüntülerini hizalamak için)
align_to = rs.stream.color
align = rs.align(align_to)
classNames = {39: "bottle"}  # Add other class mappings as needed

model = YOLO("yolov8l-seg.pt")

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


        color_intrinsics = aligned_frames.get_color_frame().get_profile().as_video_stream_profile().get_intrinsics()
        focal_length = color_intrinsics.fx
        print(focal_length)


        # YOLO modelini kullanarak şişeyi tespit etme
        results = model(color_image, stream=True, conf=0.5, save=False, show=True, save_txt=False, verbose=False, classes=39)
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
                    # print("piksel_w:",w)
                    # print("piksel_h:",h)
                    cvzone.cornerRect(color_image, (x1, y1, w, h), colorR=(255, 255, 0), colorC=(255, 0, 0))

                    # Şişenin orta noktasını hesaplama
                    x_mid = int((x1 + x2) / 2)
                    y_mid = int((y1 + y2) / 2)

                    # Derinlik bilgisini alma
                    depth_value = depth_frame.get_distance(x_mid, y_mid) *100
                    # print(f"Şişenin Orta Noktası: ({x_mid}, {y_mid}), Derinlik: {depth_value:.2f} cm")


                    width_cm = ((w * depth_value) / focal_length)
                    height_cm = ((h * depth_value) / focal_length)

                    # Bilgileri Görüntüye Yazdırma
                    cvzone.putTextRect(color_image, f'W: {width_cm:.1f}cm, H: {height_cm:.1f}cm',
                                   (x1, y1 - 40),  # x1, y1 - 40
                                   scale=1.5,
                                   colorR=(0, 0, 0))


                    # Bilgileri Görüntüye Yazdırma
                    cv2.circle(color_image, (x_mid, y_mid), 5, (0, 255, 0), 2)
                    cvzone.putTextRect(color_image, f'Depth: {depth_value:.2f}cm', (x_mid, y_mid - 10), scale=1.5, colorR=(0, 255, 0))

        # Görüntüyü göster
        cv2.imshow("Image Detector", color_image)
        if cv2.waitKey(30) == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()