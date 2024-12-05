import cv2
import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO
import torch
from cvzone import cornerRect, putTextRect
import sqlite3
import time

# Sabit hat derinliği (cm)
reference_depth = 71  # Hattın referans derinliği
blur_ratio = 50

class VolumeMeasurement:
    def __init__(self):
        # RealSense pipeline başlatma
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

        self.pipeline_active = False
        self.running = False  # Initialize the running status

        # Kameranın intrinsics bilgilerini al
        profile = self.pipeline.start(self.config)
        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.focal_length = intrinsics.fx  # Odak uzaklığı (piksel cinsinden)
        self.cx, self.cy = intrinsics.ppx, intrinsics.ppy  # Görüntü merkez noktası

        # YOLO segmentasyon modeli yükleme (GPU desteğiyle)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO("yolov11.pt").to(device)  # Modeli GPU'ya yükle

        # Sınıf renkleri
        self.colors = {"Hat": (255, 0, 0), "Paket": (153, 204, 255)}

        # Frame ve veritabanı için değişkenler
        self.frame_id = 0

        # FPS ve zaman bilgisi
        self.start_time = time.time()
        self.frames_processed = 0
        self.prev_time = self.start_time  # For FPS calculation

        # Veritabanı bağlantısı
        self.conn = sqlite3.connect("measurements.db", check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        """Veritabanı tablosunu oluşturur."""
        self.cursor.execute(""" 
        CREATE TABLE IF NOT EXISTS measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            frame_id INTEGER,
            class_name TEXT,
            volume REAL,
            area REAL,
            width REAL,
            height REAL,
            depth REAL,
            confidence REAL,
            bbox_coords TEXT,
            timestamp TEXT,
            fps REAL
        )
        """)
        self.conn.commit()

    def start_pipeline(self):
        """Kamerayı başlatır.""" 
        try: 
            if not self.pipeline_active: 
                try: 
                    self.pipeline.stop() 
                except Exception: 
                    pass 
                self.pipeline.start(self.config) 
                self.pipeline_active = True 
                self.running = True  # Camera is running now
        except Exception as e: 
            print(f"Pipeline başlatılırken hata oluştu: {e}") 

    def stop_pipeline(self): 
        """Kamerayı durdurur.""" 
        try: 
            if self.pipeline_active: 
                self.pipeline.stop() 
                self.pipeline_active = False
                self.running = False  # Camera is no longer running
        except Exception as e: 
            print(f"Pipeline durdurulurken hata oluştu: {e}") 

    def capture_frame(self): 
        """Kameradan frame alır ve işlenmiş frame döner.""" 
        if not self.pipeline_active: 
            return None

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=10000)  # 10 saniye timeout
        except Exception as e:
            print(f"Frame alınamadı: {e}")
            return None

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            print("Kamera verisi alınamadı")
            return None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_scale = depth_frame.get_units() * 100  # mm -> cm

        annotated_frame = self.process_frame(color_image, depth_image, depth_scale, self.focal_length, reference_depth)
        return annotated_frame

    def process_frame(self, color_image, depth_image, depth_scale, focal_length, reference_depth):
        """Frame'i işler ve veritabanına kaydeder."""
        self.frame_id += 1  # Çerçeve ID'sini artır
        start_time = time.time()

        results = self.model(color_image, stream=True, conf=0.90)
        annotated_frame = color_image.copy()

        detected = False  # Tespit kontrolü

        for r in results:
            if r.masks:
                detected = True
                for mask, box, cls_id, conf in zip(r.masks.data, r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                    class_name = self.model.names[int(cls_id)]
                    confidence = float(conf.cpu().numpy())  # Doğru şekilde güven puanını alın

                    if class_name == "Paket":
                        mask_np = mask.cpu().numpy()
                        mask_resized = cv2.resize(mask_np, (color_image.shape[1], color_image.shape[0]))

                        x_min, y_min, x_max, y_max = box.cpu().numpy()
                        contours, _ = cv2.findContours((mask_resized > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            largest_contour = max(contours, key=cv2.contourArea)
                            real_width, real_height, box = self.calculate_dimensions(largest_contour, depth_image, depth_scale, focal_length)
                            real_area, depth_difference, package_volume = self.calculate_volume_with_reference(
                                largest_contour, depth_image, depth_scale, reference_depth
                            )

                            # Alan hesabı için
                            cv2.drawContours(annotated_frame, [box], -1, self.colors[class_name], 1)
                            x_min = np.min(box[:, 0])
                            x_max = np.max(box[:, 0])
                            y_min = np.min(box[:, 1])
                            y_max = np.max(box[:, 1])
                            x_mid = int((x_min + x_max) / 2)
                            y_mid = int((y_min + y_max) / 2)
                            w = int(x_max - x_min)
                            h = int(y_max - y_min)
                            mask_indices = (mask_resized > 0).astype(np.uint8)
                            blurred_frame = cv2.blur(annotated_frame, (blur_ratio, blur_ratio))
                            annotated_frame[mask_indices == 1] = blurred_frame[mask_indices == 1]
                            
                            # En dış kutu
                            cornerRect(annotated_frame, (x_min, y_min, w, h), l=15, t=3, colorC=(153, 204, 255))

                            info_text = f'Hacim: {package_volume:.2f} cm3 | Alan: {real_area:.2f} cm2'
                            coord_text = f'W:{real_width:.2f}cm | H:{real_height:.2f}cm | D:{depth_difference:.2f}cm'

                            putTextRect(annotated_frame, info_text,
                                        (x_min, y_min - 45), scale=0.9, thickness=1,
                                        colorR=(153, 255, 153), colorT=(0, 0, 0), offset=5)
                            putTextRect(annotated_frame, coord_text,
                                        (x_min, y_min - 25), scale=0.9, thickness=1,
                                        colorR=(153, 255, 153), colorT=(0, 0, 0), offset=5)

                            cv2.circle(annotated_frame, (x_mid, y_mid), 5, (255, 153, 255), -1)

                            # Zaman bilgisi ve FPS hesaplaması
                            elapsed_time = time.time() - self.prev_time
                            self.frames_processed += 1

                            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')  # timestamp her durumda alınacak şekilde buraya taşıyoruz

                            # FPS hesaplaması
                            fps = 0  # Başlangıç değeri
                            if elapsed_time >= 1.0:
                                fps = self.frames_processed / elapsed_time
                                self.frames_processed = 0
                                self.prev_time = time.time()
                                print(f"FPS: {fps:.2f}")

                            # Veritabanı kaydı
                            self.cursor.execute("""
                                INSERT INTO measurements (frame_id, class_name, volume, area, width, height, depth, 
                                     confidence, bbox_coords, timestamp, fps)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ? )
                            """, (
                                self.frame_id, class_name, package_volume, real_area, real_width, real_height, depth_difference,
                                confidence, f"({x_min},{y_min},{x_max},{y_max})", timestamp, fps
                            ))
                            self.conn.commit()

        if not detected:
            print("Hiçbir obje tespit edilmedi.")
        
        return annotated_frame

    def calculate_volume_with_reference(self, contour, depth_frame, depth_scale_cm, reference_depth):
        mask = np.zeros_like(depth_frame, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        masked_depth = depth_frame[mask > 0]

        if len(masked_depth) == 0:
            return 0, 0, 0

        avg_depth = np.mean(masked_depth) * depth_scale_cm
        depth_difference = reference_depth - avg_depth + 1
        pixel_area = cv2.contourArea(contour)
        real_area = (pixel_area * (avg_depth ** 2)) / (self.focal_length ** 2)
        volume = real_area * abs(depth_difference)
        return real_area, depth_difference, volume

    def calculate_dimensions(self, contour, depth_frame, depth_scale_cm, focal_length):
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        width, height = rect[1]
        if width < height:
            width, height = height, width

        mask = np.zeros_like(depth_frame, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        masked_depth = depth_frame[mask > 0]
        avg_depth = np.mean(masked_depth) * depth_scale_cm

        real_width = (width * avg_depth) / focal_length
        real_height = (height * avg_depth) / focal_length

        return real_width, real_height, box
