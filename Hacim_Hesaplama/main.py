import sys
import os
import csv
import psutil
import subprocess
from PyQt6.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsEllipseItem, QGraphicsTextItem, QFileDialog
from PyQt6.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap, QBrush, QColor, QPen, QFont
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from interface import Ui_MainWindow  
import cv2
import numpy as np
from hacim import *
from pyqtgraph import PlotItem, GraphicsLayoutWidget,BarGraphItem
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter



class VolumeMeasurementThread(QThread):
    """VolumeMeasurement sınıfını ayrı bir iş parçacığında çalıştırır."""
    frame_ready = pyqtSignal(np.ndarray)  # Görüntü gönderimi için sinyal

    def __init__(self):
        super().__init__()
        self.volume_measurement = VolumeMeasurement()
        self._running = False

    def run(self):
        """İş parçacığı başlatıldığında sürekli kare yakalar."""
        self.volume_measurement.start_pipeline()
        self._running = True

        while self._running:
            frame = self.volume_measurement.capture_frame()
            if frame is not None:
                self.frame_ready.emit(frame)  # Görüntüyü sinyal ile gönder

    def stop(self):
        """İş parçacığını durdurur."""
        self._running = False
        self.volume_measurement.stop_pipeline()
        self.wait()  # İş parçacığı tamamen bitene kadar bekler



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Veritabanı bağlantısı
        self.conn = sqlite3.connect("measurements.db")
        self.cursor = self.conn.cursor()

        # QThread ve Kamera İşlemleri
        self.thread = VolumeMeasurementThread()
        self.thread.frame_ready.connect(self.update_frame)

        # Butonlara işlev ekleme
        self.ui.toolButton_start_9.clicked.connect(self.start_camera)
        self.ui.toolButton_stop_5.clicked.connect(self.stop_camera)
        self.ui.toolButton_pdf_5.clicked.connect(self.export_pdf)
        self.ui.toolButton_csv.clicked.connect(self.export_csv)


        # Grafik Butonları
        self.setup_line_chart()
        self.setup_pie_chart()
        self.setup_bar_chart()


        # toolButton_line'a tıklama olayı bağla
        self.ui.toolButton_line.clicked.connect(self.start_dynamic_chart)
        self.ui.toolButton_pie.clicked.connect(self.start_dynamic_pie_chart)
        self.ui.toolButton_bar.clicked.connect(self.start_bar_chart)
        self.ui.toolButton_save.clicked.connect(self.save_all_graphics)


        # Timer
        self.line_chart_timer=None

        # Grafik sahneleri
        self.cpu_scene = QGraphicsScene(self)
        self.gpu_scene = QGraphicsScene(self)

        # GrafikView widget'larına sahneleri bağlama
        self.ui.graphicsView_cpu.setScene(self.cpu_scene)
        self.ui.graphicsView_gpu.setScene(self.gpu_scene)

        # Timer ile sistem durumu güncelleme
        self.usage_timer = QTimer(self)
        self.usage_timer.timeout.connect(self.update_usage)
        self.usage_timer.start(1000)  # Her saniye güncelleme

        # Başlangıçta göstergeyi sıfırla
        self.update_donut_chart(self.cpu_scene, 0)
        self.update_donut_chart(self.gpu_scene, 0)
        self.ui.progressBar_cpu.setValue(0)
        self.ui.progressBar_gpu.setValue(0)

        self.show()

        self.exporting = False

    def start_camera(self):
        """Kamerayı başlat."""  
        if not self.thread.isRunning():
            self.thread.start()
            self.ui.toolButton_start_9.setEnabled(False)
            self.ui.toolButton_stop_5.setEnabled(True)
            print("Kamera Başlatıldı")  # Bu satır sadece debug amacıyla eklenmiştir

    def stop_camera(self):
        """Kamerayı durdur."""
        if self.thread.isRunning():
            self.thread.stop()
            self.ui.toolButton_start_9.setEnabled(True)
            self.ui.toolButton_stop_5.setEnabled(False)
            print("Kamera Durduruldu")  # Bu satır sadece debug amacıyla eklenmiştir
    def update_frame(self, frame):
        """QThread'den gelen görüntüyü günceller."""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV BGR formatını RGB'ye dönüştürür
        height, width, channel = frame.shape
        q_image = QImage(frame.data, width, height, channel * width, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.ui.label_cam.setPixmap(pixmap)  # Görüntüyü QLabel üzerine yerleştir

    def export_pdf(self):
        """Veritabanındaki veriyi PDF'ye aktar."""
        if self.exporting:
            return

        self.exporting = True
        self.ui.toolButton_pdf_5.setEnabled(False)

        # PDF dosyasını hazırlıyoruz
        self.pdf_rows = []
        conn = sqlite3.connect("measurements.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM measurements")
        self.pdf_rows = cursor.fetchall()

        # PDF çıktısı oluşturma
        self.file_path = "measurements_report.pdf"
        self.c = canvas.Canvas(self.file_path, pagesize=letter)
        self.c.setFont("Helvetica-Bold", 16)
        self.c.drawString(100, 750, "Measurements Report")
        self.c.setFont("Helvetica-Bold", 10)
        self.c.drawString(100, 730, "ID | Frame ID | Class Name | Volume | Area | Width | Height | Depth | Confidence | BBox Coords | Timestamp | FPS")
        self.y_position = 710

    def export_csv(self):
        """Veritabanındaki verileri CSV'ye aktar."""
        file_name = "measurements_report.csv"
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["ID", "Frame ID", "Class Name", "Volume", "Area", "Width", "Height", "Depth", "Confidence", "BBox Coords", "Timestamp", "FPS"])
            self.cursor.execute("SELECT * FROM measurements")
            rows = self.cursor.fetchall()
            writer.writerows(rows)
        print(f"CSV file created: {file_name}")

    def update_donut_chart(self, scene, value):
        """Donut grafiğini günceller ve yüzdeyi gösterir."""
        scene.clear()
        center_x, center_y = 100, 100
        radius = 80
        thickness = 30

        # Dış çember (donut grafiği)
        ellipse = QGraphicsEllipseItem(center_x - radius, center_y - radius, 2 * radius, 2 * radius)
        ellipse.setBrush(QBrush(QColor("#d3d3d3")))
        ellipse.setPen(QPen(Qt.PenStyle.NoPen))
        scene.addItem(ellipse)

        # Yüzdeler
        if value > 0:
            angle_span = int((value / 100) * 360 * 16)
            pie = QGraphicsEllipseItem(center_x - radius, center_y - radius, 2 * radius, 2 * radius)
            pie.setBrush(QBrush(QColor("#4caf50")))
            pie.setPen(QPen(Qt.PenStyle.NoPen))
            pie.setStartAngle(90 * 16)
            pie.setSpanAngle(-angle_span)
            scene.addItem(pie)

        # İç çember
        inner_ellipse = QGraphicsEllipseItem(center_x - radius + thickness, center_y - radius + thickness,
                                             2 * (radius - thickness), 2 * (radius - thickness))
        inner_ellipse.setBrush(QBrush(QColor("#ffffff")))
        inner_ellipse.setPen(QPen(Qt.PenStyle.NoPen))
        scene.addItem(inner_ellipse)

        # Yüzdelik metin
        percentage_text = QGraphicsTextItem(f"{value:.0f}%")
        percentage_text.setDefaultTextColor(QColor("#000000"))
        percentage_text.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        percentage_text.setPos(center_x - 30, center_y - 20)
        scene.addItem(percentage_text)

    def get_memory_usage(self):
        """RAM kullanımını döndürür."""
        return psutil.virtual_memory().percent

    def get_disk_usage(self):
        """Disk kullanımını döndürür."""
        return psutil.disk_usage('/').percent

    def get_cpu_usage(self):
        """CPU kullanımını döndürür."""
        return psutil.cpu_percent(interval=0.1)

    def get_gpu_usage(self):
        """GPU kullanımını döndürür."""
        try:
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                encoding='utf-8'
            )
            return float(result.strip())
        except subprocess.CalledProcessError as e:
            print(f"GPU kullanımı alınamadı: {e}")
            return 0
        except Exception as e:
            print(f"Bir hata oluştu: {e}")
            return 0

    def update_usage(self):
        """CPU, GPU, RAM ve disk kullanımını günceller."""
        cpu_usage = self.get_cpu_usage()
        gpu_usage = self.get_gpu_usage()
        memory_usage = self.get_memory_usage()
        disk_usage = self.get_disk_usage()

        self.update_donut_chart(self.cpu_scene, cpu_usage)
        self.update_donut_chart(self.gpu_scene, gpu_usage)
        self.ui.progressBar_cpu.setValue(int(memory_usage))
        self.ui.progressBar_gpu.setValue(int(disk_usage))

    def setup_line_chart(self):
        """graphicsView_line için grafik alanı oluşturur."""
        # PyQtGraph GraphicsLayoutWidget kullanarak grafik sahnesi oluştur
        self.line_chart_widget = GraphicsLayoutWidget()
        scene = QGraphicsScene(self)
        self.ui.graphicsView_line.setScene(scene)
        scene.addWidget(self.line_chart_widget)

        # Grafik alanını ekle
        self.line_chart = self.line_chart_widget.addPlot()
        self.line_chart.setTitle("FPS Değerleri", color="b", size="12pt")
        self.line_chart.setLabel('left', 'FPS', color='blue', size='10pt')
        self.line_chart.setLabel('bottom', 'Frame', color='blue', size='10pt')
        self.line_chart.showGrid(x=True, y=True)

    def get_fps_data(self):
        """Veritabanından FPS verilerini çeker."""
        try:
            self.cursor.execute("SELECT fps FROM measurements WHERE fps > 0")
            rows = self.cursor.fetchall()
            return [row[0] for row in rows if row[0]>0]  # FPS verilerini liste olarak döndür
        except sqlite3.Error as e:
            print(f"Veritabanı hatası: {e}")
            return []

    def draw_line_chart(self):
        """Grafiği bir kez çizer."""
        fps_data = self.get_fps_data()  # Veritabanından FPS verilerini al
        if fps_data:  # Eğer veri varsa
            x = list(range(len(fps_data)))  # X ekseni için frame indexleri
            self.line_chart.clear()  # Önceki grafiği temizle
            self.line_chart.plot(x, fps_data, pen='b')  # Yeni grafik çiz
        else:
            print("FPS verisi bulunamadı, grafik çizilemiyor.")

    def update_line_chart(self):
        """Grafiği dinamik olarak günceller."""
        self.draw_line_chart()  # Grafik her güncellemede yeniden çizilir

    def start_dynamic_chart(self):
        """Dinamik grafik güncellemeyi başlatır."""
        # stackedWidget içinde doğru sayfaya geç
        self.ui.stackedWidget.setCurrentWidget(self.ui.line)

        # Grafik bir kez çizilir
        self.draw_line_chart()

        # Dinamik güncelleme için QTimer başlatılır
        if not self.line_chart_timer:
            self.line_chart_timer = QTimer(self)
            self.line_chart_timer.timeout.connect(self.update_line_chart)
            self.line_chart_timer.start(1000)  # Her 1 saniyede bir güncelleme


    def get_confidence_data(self):
        """Veritabanından Confidence değerlerini çeker."""
        try:
            self.cursor.execute("SELECT Confidence FROM measurements WHERE Confidence IS NOT NULL")
            rows = self.cursor.fetchall()
            return [row[0] for row in rows]  # Confidence değerlerini liste olarak döndür
        except sqlite3.Error as e:
            print(f"Veritabanı hatası: {e}")
            return []

    def setup_pie_chart(self):
        """graphicsView_pie için grafik alanı oluşturur."""
        self.pie_chart_widget = GraphicsLayoutWidget()
        scene = QGraphicsScene(self)
        self.ui.graphicsView_pie.setScene(scene)
        scene.addWidget(self.pie_chart_widget)

        self.pie_chart = self.pie_chart_widget.addPlot()
        self.pie_chart.setTitle("Confidence Değerleri", color="b", size="12pt")
        self.pie_chart.setLabel('left', 'Confidence', color='blue', size='10pt')
        self.pie_chart.setLabel('bottom', 'Frame', color='blue', size='10pt')
        self.pie_chart.showGrid(x=True, y=True)

    def draw_pie_chart(self):
        """graphicsView_pie içine Confidence verilerini çizer."""
        confidence_data = self.get_confidence_data()
        if confidence_data:
            x = list(range(len(confidence_data)))
            self.pie_chart.clear()
            self.pie_chart.plot(x, confidence_data, pen='g')
        else:
            print("Confidence verisi bulunamadı, grafik çizilemiyor.")

    def update_pie_chart(self):
        """Dinamik olarak Confidence grafiğini günceller."""
        self.draw_pie_chart()

    def start_dynamic_pie_chart(self):
        """Dinamik Confidence grafik güncellemeyi başlatır."""
        self.ui.stackedWidget.setCurrentWidget(self.ui.graphicsView_pie.parentWidget())
        self.draw_pie_chart()
        if not hasattr(self, "pie_chart_timer"):
            self.pie_chart_timer = QTimer(self)
            self.pie_chart_timer.timeout.connect(self.update_pie_chart)
            self.pie_chart_timer.start(1000)

    def setup_bar_chart(self):
        """graphicsView_bar için bar plot alanını ayarlar."""
        # PyQtGraph GraphicsLayoutWidget kullanımı
        self.bar_chart_widget = GraphicsLayoutWidget()
        scene = QGraphicsScene(self)
        self.ui.graphicsView_bar.setScene(scene)
        scene.addWidget(self.bar_chart_widget)

        # Bar plot alanı oluştur
        self.bar_chart = self.bar_chart_widget.addPlot()
        self.bar_chart.setTitle("Ortalama FPS ve MAP Değerleri", color="b", size="12pt")
        self.bar_chart.setLabel('left', 'Ortalama Değer', color='blue', size='10pt')
        self.bar_chart.setLabel('bottom', '', color='blue', size='10pt')
        self.bar_chart.showGrid(x=True, y=True)

    def get_average_values(self):
        """Veritabanından ortalama FPS ve MAP değerlerini çeker."""
        try:
            # Ortalama FPS ve Confidence (MAP) değerlerini SQL ile çek
            self.cursor.execute("SELECT AVG(fps), AVG(Confidence) FROM measurements")
            result = self.cursor.fetchone()
            if result:
                return result[0], result[1]  # Ortalama FPS ve MAP döndür
            return 0, 0
        except sqlite3.Error as e:
            print(f"Veritabanı hatası: {e}")
            return 0, 0

    def draw_bar_chart(self):
        """graphicsView_bar üzerinde bar plot çizer."""
        avg_fps, avg_map = self.get_average_values()  # Ortalama FPS ve MAP değerlerini al

        # Bar verilerini ayarla
        x = [0, 1]  # X ekseni: 0 -> FPS, 1 -> MAP
        y = [avg_fps, avg_map]  # Y ekseni: Ortalama FPS ve MAP
        bar_width = 0.4  # Bar genişliği

        # Bar grafiği temizle ve yeni verileri çiz
        self.bar_chart.clear()
        bars = BarGraphItem(x=x, height=y, width=bar_width, brush='b')  # Mavi barlar
        self.bar_chart.addItem(bars)

        # X ekseni etiketlerini ekle
        self.bar_chart.getAxis('bottom').setTicks([[(0, 'FPS'), (1, 'MAP')]])

    def start_bar_chart(self):
        """toolButton_bar tıklanınca bar chart çizer ve ilgili sayfayı gösterir."""
        # stackedWidget içinde doğru sayfaya geçiş yap
        self.ui.stackedWidget.setCurrentWidget(self.ui.graphicsView_bar.parentWidget())

        # Bar plot çiz
        self.draw_bar_chart()

    def save_all_graphics(self):
        """
        Pie Chart, Bar Plot ve Line Chart'ı seçilen klasöre PNG olarak kaydeder.
        """
        # Kullanıcıdan bir dosya konumu seçmesini iste
        folder_path = QFileDialog.getExistingDirectory(self, "Grafikleri Kaydetmek İçin Bir Klasör Seç")

        if folder_path:  # Eğer kullanıcı bir klasör seçtiyse
            try:
                # Pie Chart'ı kaydet
                pie_file_path = os.path.join(folder_path, "pie_chart.png")
                exporter = pg.exporters.ImageExporter(self.pie_chart_widget.scene())
                exporter.export(pie_file_path)
                print(f"Pie Chart kaydedildi: {pie_file_path}")

                # Bar Plot'u kaydet
                bar_file_path = os.path.join(folder_path, "bar_plot.png")
                exporter = pg.exporters.ImageExporter(self.bar_chart_widget.scene())
                exporter.export(bar_file_path)
                print(f"Bar Plot kaydedildi: {bar_file_path}")

                # Line Chart'ı kaydet
                line_file_path = os.path.join(folder_path, "line_chart.png")
                exporter = pg.exporters.ImageExporter(self.line_chart_widget.scene())
                exporter.export(line_file_path)
                print(f"Line Chart kaydedildi: {line_file_path}")

            except Exception as e:
                print(f"Hata oluştu: {e}")
        else:
            print("Kullanıcı bir dosya konumu seçmedi.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())
