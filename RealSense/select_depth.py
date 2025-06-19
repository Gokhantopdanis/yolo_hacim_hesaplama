import cv2
import pyrealsense2 as rs
import numpy as np

# RealSense pipeline başlatma
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# Hizalayıcı oluşturma (derinlik ve renk görüntülerini hizalamak için)
align_to = rs.stream.color
align = rs.align(align_to)

# İşaretlenen noktanın koordinatlarını saklamak için değişken
point = (0, 0)
depth_value = 0

def mouse_callback(event, x, y, flags, param):
    global point, depth_value
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        depth_value = depth_frame.get_distance(x, y)
        print(f"Tıklanan Nokta: {point}, Derinlik: {depth_value:.2f} cm")

# Pencere ve fare geri çağırma fonksiyonu oluşturma
cv2.namedWindow("Image Detector")
cv2.setMouseCallback("Image Detector", mouse_callback)

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

        # İşaretlenen noktadaki derinlik bilgisini görselleştirme
        cv2.circle(color_image, point, 5, (0, 255, 0), 2)
        cv2.putText(color_image, f'Depth: {depth_value:.2f}cm', (point[0], point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Görüntüyü göster
        cv2.imshow("Image Detector", color_image)
        if cv2.waitKey(30) == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()