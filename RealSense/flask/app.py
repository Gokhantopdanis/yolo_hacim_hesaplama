import sys
import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
import pyrealsense2 as rs
from ultralytics import YOLO
import cvzone


class CameraThread(QThread):
    update_frame = pyqtSignal(np.ndarray, dict)

    def __init__(self):
        super().__init__()
        self.is_running = False
        self.model = YOLO("yolov8l-seg.pt")
        self.classNames = {39: "bottle"}
        self.pipeline = rs.pipeline()
        self.align = None

    def init_camera(self):
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

    def run(self):
        self.is_running = True
        self.init_camera()

        while self.is_running:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            focal_length = color_intrinsics.fx

            results = self.model(color_image, stream=True, conf=0.51, classes=39)
            data = {}
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    cls_name = self.classNames.get(cls, "unknown")
                    if cls_name == "bottle":
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        w, h = x2 - x1, y2 - y1
                        x_mid, y_mid = (x1 + x2) // 2, (y1 + y2) // 2
                        depth_value = depth_frame.get_distance(x_mid, y_mid) * 100
                        width_cm = (w * depth_value) / focal_length
                        height_cm = (h * depth_value) / focal_length
                        data = {
                            "width": round(width_cm, 1),
                            "height": round(height_cm, 1),
                            "depth": round(depth_value, 2),
                            "x_mid": x_mid,
                            "y_mid": y_mid
                        }

                        # Sarı köşe ve kenar efektleri
                        cvzone.cornerRect(color_image, (x1, y1, w, h), l=15, t=3, colorR=(0, 255, 255),
                                          colorC=(255, 255, 0))

                        # Yazı efektleri
                        info_text = f'Size: W:{width_cm:.1f}cm | H:{height_cm:.1f}cm | D:{depth_value:.2f}cm'
                        coord_text = f'Center: ({x_mid}, {y_mid})'
                        cvzone.putTextRect(color_image, info_text,
                                           (x1, y1 - 35), scale=0.7, thickness=2,
                                           colorR=(50, 50, 50), colorT=(0, 255, 0), offset=5)
                        cvzone.putTextRect(color_image, coord_text,
                                           (x1, y1 - 15), scale=0.7, thickness=2,
                                           colorR=(50, 50, 50), colorT=(0, 255, 0), offset=5)

                        # Orta nokta işareti
                        cv2.circle(color_image, (x_mid, y_mid), 5, (0, 255, 0), -1)

            self.update_frame.emit(color_image, data)

    def stop(self):
        self.is_running = False
        self.pipeline.stop()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Object Detection")
        self.setStyleSheet("background-color: #1e1e1e; color: #ffffff;")
        self.camera_thread = CameraThread()

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.info_label = QLabel("Width: - cm | Height: - cm | Depth: - cm")
        self.info_label.setFont(QFont("Arial", 12))
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_camera)
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_camera)
        self.stop_button.setStyleSheet("background-color: #f44336; color: white; padding: 10px;")

        h_layout = QHBoxLayout()
        h_layout.addWidget(self.start_button)
        h_layout.addWidget(self.stop_button)

        v_layout = QVBoxLayout()
        v_layout.addWidget(self.video_label)
        v_layout.addWidget(self.info_label)
        v_layout.addLayout(h_layout)

        self.setLayout(v_layout)

        # Kamera akışını güncelleme sinyali
        self.camera_thread.update_frame.connect(self.update_image)

    def start_camera(self):
        self.camera_thread.start()

    def stop_camera(self):
        self.camera_thread.stop()

    def update_image(self, frame, data):
        # Video akışını QLabel üzerinde göster
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

        # Nesne tespiti bilgilerini güncelleme
        self.info_label.setText(
            f"Width: {data.get('width', '-')} cm | Height: {data.get('height', '-')} cm | Depth: {data.get('depth', '-')} cm")

    def closeEvent(self, event):
        # Pencere kapanırken kamera iş parçacığını durdurma
        self.camera_thread.stop()
        self.camera_thread.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
