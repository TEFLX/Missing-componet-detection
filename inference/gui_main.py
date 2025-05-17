# gui_app.py (final full version with batch visualization)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget,
                             QTextEdit, QHBoxLayout, QTabWidget, QMessageBox, QListWidget, QStackedWidget, QListWidgetItem)
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QPalette
from PyQt5.QtCore import Qt
from ultralytics import YOLO
import cv2
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔍 Missing Component Detection Suite")
        self.setFixedSize(1400, 850)

        self.model = YOLO("best.pt")
        self.expected_components = ['baseplate', 'childpart1', 'childpart2', 'clinching1', 'pin1', 'pin2']
        self.last_results = None
        self.last_image = None

        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; color: #ffffff; }
            QLabel#TitleLabel { font-size: 28px; font-weight: bold; padding: 20px; color: #ffffff; }
            QPushButton {
                background-color: #3e8e41; color: white; padding: 10px;
                font-size: 16px; border-radius: 8px;
            }
            QPushButton:hover { background-color: #2e7031; }
            QTextEdit {
                background-color: #252526; color: #ffffff; font-family: Consolas;
                font-size: 14px; padding: 10px; border: 1px solid #3c3c3c;
                border-radius: 5px;
            }
            QListWidget {
                background-color: #2d2d30; color: #ffffff; font-size: 16px; border: none;
            }
            QListWidget::item:selected {
                background-color: #007acc; color: #ffffff;
            }
        """)

        self.init_ui()

    def init_ui(self):
        container = QWidget()
        layout = QHBoxLayout(container)

        self.nav = QListWidget()
        self.nav.addItem("🔍 Detector")
        self.nav.addItem("📄 Export & Summary")
        self.nav.addItem("🖼 Batch Processing")
        self.nav.addItem("📘 Info")
        self.nav.setFixedWidth(200)
        self.nav.currentRowChanged.connect(self.display_page)

        self.stack = QStackedWidget()
        self.stack.addWidget(self.build_detector_tab())
        self.stack.addWidget(self.build_export_tab())
        self.stack.addWidget(self.build_batch_tab())
        self.stack.addWidget(self.build_info_tab())

        layout.addWidget(self.nav)
        layout.addWidget(self.stack)
        self.setCentralWidget(container)
        self.nav.setCurrentRow(0)

    def display_page(self, index):
        self.stack.setCurrentIndex(index)

    def build_detector_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        self.title_label = QLabel("🧠 Component Detection")
        self.title_label.setObjectName("TitleLabel")
        self.title_label.setAlignment(Qt.AlignCenter)

        self.image_label = QLabel("\n\n Drag or browse to upload image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed #444; background-color: #2d2d30;")
        self.image_label.setFixedHeight(400)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFixedHeight(200)

        browse_btn = QPushButton("📂 Browse Image")
        browse_btn.clicked.connect(self.load_image)

        save_btn = QPushButton("💾 Save Annotated Image")
        save_btn.clicked.connect(self.save_annotated_image)

        webcam_btn = QPushButton("🎥 Run Webcam")
        webcam_btn.clicked.connect(self.run_webcam)

        reset_btn = QPushButton("🔄 Reset")
        reset_btn.clicked.connect(self.reset_gui)

        btns = QHBoxLayout()
        btns.addWidget(browse_btn)
        btns.addWidget(save_btn)
        btns.addWidget(webcam_btn)
        btns.addWidget(reset_btn)

        layout.addWidget(self.title_label)
        layout.addWidget(self.image_label)
        layout.addLayout(btns)
        layout.addWidget(self.result_text)
        tab.setLayout(layout)
        return tab

    def build_export_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        export_btn = QPushButton("📄 Export Results to Excel")
        export_btn.clicked.connect(self.export_results)

        summary_btn = QPushButton("📊 Generate Summary Chart")
        summary_btn.clicked.connect(self.show_summary_chart)

        layout.addWidget(export_btn)
        layout.addWidget(summary_btn)
        tab.setLayout(layout)
        return tab

    def build_batch_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        batch_btn = QPushButton("🖼 Batch Process Folder")
        batch_btn.clicked.connect(self.batch_process)

        visual_btn = QPushButton("📈 Visualize Batch Report")
        visual_btn.clicked.connect(self.visualize_batch_report)

        layout.addWidget(batch_btn)
        layout.addWidget(visual_btn)
        tab.setLayout(layout)
        return tab

    def build_info_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        info = QTextEdit()
        info.setReadOnly(True)
        info.setText("""
This PyQt5 application detects missing components using a YOLOv8 model.

• Upload an image
• Detect parts
• See what's missing
• Export results
• Run webcam & batch mode
• Generate summary charts
        """)
        layout.addWidget(info)
        tab.setLayout(layout)
        return tab

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.last_image = file_path
            self.run_detection(file_path)

    def run_detection(self, image_path):
        results = self.model(image_path)
        detected_classes = [self.model.names[int(cls)] for cls in results[0].boxes.cls]
        missing = list(set(self.expected_components) - set(detected_classes))
        self.result_text.setPlainText(f"✅ Detected: {detected_classes}\n❌ Missing: {missing}")
        self.last_results = {'Detected': detected_classes, 'Missing': missing, 'Results': results}

        annotated_img = results[0].plot()
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        h, w, ch = annotated_img.shape
        qt_image = QImage(annotated_img.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(1280, 400, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

    def save_annotated_image(self):
        if self.last_results:
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "annotated.jpg", "Images (*.jpg *.png)")
            if save_path:
                annotated_img = self.last_results['Results'][0].plot()
                cv2.imwrite(save_path, annotated_img)
                QMessageBox.information(self, "Saved", f"Image saved to:\n{save_path}")
        else:
            QMessageBox.warning(self, "No Results", "Run detection first.")

    def export_results(self):
        if self.last_results:
            detected = [c for c in self.expected_components if c in self.last_results['Detected']]
            missing = [c for c in self.expected_components if c in self.last_results['Missing']]
            df = pd.DataFrame({'Component': self.expected_components})
            df['Detected'] = df['Component'].apply(lambda x: '✔️' if x in detected else '')
            df['Missing'] = df['Component'].apply(lambda x: '❌' if x in missing else '')
            save_path, _ = QFileDialog.getSaveFileName(self, "Export to Excel", "results.xlsx", "Excel Files (*.xlsx)")
            if save_path:
                df.to_excel(save_path, index=False)
                QMessageBox.information(self, "Exported", f"Results exported to:\n{save_path}")
        else:
            QMessageBox.warning(self, "No Results", "Run detection first.")

    def run_webcam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.critical(self, "Camera Error", "Could not open webcam.")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model.predict(source=frame, save=False, show=True)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def reset_gui(self):
        self.image_label.setPixmap(QPixmap())
        self.result_text.clear()
        self.last_results = None
        self.last_image = None

    def batch_process(self):
        import qrcode
        from datetime import datetime
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            summary = []
            image_id = 1
            date_code = datetime.now().strftime("%d%m%Y")
            product_name = "MC"
            qr_folder = os.path.join(folder, "qr_codes")
            os.makedirs(qr_folder, exist_ok=True)
            for f in os.listdir(folder):
                if f.lower().endswith(('png', 'jpg', 'jpeg')):
                    path = os.path.join(folder, f)
                    results = self.model(path)
                    detected = [self.model.names[int(cls)] for cls in results[0].boxes.cls]
                    missing = list(set(self.expected_components) - set(detected))
                    img_id = f"{date_code}_{product_name}_{image_id:03d}"
                                        # Generate and save QR code
                    qr_path = os.path.join(qr_folder, f"{img_id}.png")
                    qrcode.make(img_id).save(qr_path)

                    summary.append({
                        'ID': img_id,
                        'Image': f,
                        'Detected': ', '.join(detected),
                        'Missing': ', '.join(missing)
                    })
                    image_id += 1
            df = pd.DataFrame(summary)
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Report", "batch_report.xlsx", "Excel Files (*.xlsx)")
            if save_path:
                df.to_excel(save_path, index=False)
                QMessageBox.information(self, "Done", f"Saved to: {save_path}")

    def show_summary_chart(self):
        if self.last_results:
            labels = ['Detected', 'Missing']
            sizes = [len(self.last_results['Detected']), len(self.last_results['Missing'])]
            colors = ['#4CAF50', '#f44336']
            plt.figure(figsize=(6, 6))
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
            plt.title('Detection Summary')
            plt.axis('equal')
            plt.show()
        else:
            QMessageBox.warning(self, "No Results", "Run detection first.")

    def visualize_batch_report(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Batch Excel", "", "Excel Files (*.xlsx)")
        if file_path:
            df = pd.read_excel(file_path)
            detected = df['Detected'].str.split(', ').apply(lambda x: len([i for i in x if i != '']))
            missing = df['Missing'].str.split(', ').apply(lambda x: len([i for i in x if i != '']))
            plt.figure(figsize=(10, 6))
            plt.bar(df['Image'], detected, label='Detected', color='#4CAF50')
            plt.bar(df['Image'], missing, bottom=detected, label='Missing', color='#f44336')
            plt.xlabel('Image File')
            plt.ylabel('Count')
            plt.title('Detected vs Missing Components per Image')
            plt.xticks(rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
