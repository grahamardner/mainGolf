import sys
import cv2

from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox, QMainWindow, QWidget, QFileDialog
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi

# frameNumberLabel


class veryCoolMan(QMainWindow):

    def __init__(self):
        super(veryCoolMan, self).__init__()
        loadUi('vidLearnMW.ui', self)

        self.startButton.clicked.connect(self.start_video)
        self.stopButton.clicked.connect(self.stop_video)
        self.nextButton.clicked.connect(self.next_frame)
        self.loadButton.clicked.connect(self.load_video)

        self.actionOpen.triggered.connect(self.load_video)
        self.actionOpen.setShortcut("Ctrl+O")
        self.actionCloseVideo.triggered.connect(self.close_video)
        self.image = None
        self.boolIsLoaded = True
        self.blankImage = None

        if self.boolIsLoaded is False:
            self.startButton.setEnabled(False)
            self.nextButton.setEnabled(False)
            self.loadButton.setEnabled(False)
            self.stopButton.setEnabled(False)

    def close_video(self):
        self.capture = None
        self.display_image(self.blankImage, 1)

    def load_video(self):

        fname, filter = QFileDialog.getOpenFileName(self, 'Open Video File',
                                                    'Z:\Programming Learning\Python\Golf',
                                                    "Video Files (*.mpeg, *.avi)")
        self.capture = cv2.VideoCapture(fname)
        intFrameNum = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.frameNumberLabel.setText(str(intFrameNum))
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

    def start_video(self):
        # self.capture = cv2.VideoCapture('p4.avi')
        # self.capture = cv2.CaptureFromeFile('p4.avi')

        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.update_frame()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000.0/30)

    def next_frame(self):
        ret, self.image = self.capture.read()

        if self.image is None:
            self.stop_video()
        else:
            self.display_image(self.image, 1)

    def update_frame(self):
        ret, self.image = self.capture.read()

        if self.image is None:
            self.stop_video()
        else:
            self.display_image(self.image, 1)

    def stop_video(self):
        self.timer.stop()

    def display_image(self, img, window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:  # [0] rows, [1] cols, [2] channels
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        # BGR >> RGB
        outImage = outImage.rgbSwapped()
        if window == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(outImage))
            self.imgLabel.setScaledContents(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = veryCoolMan()
    window.setWindowTitle('Learning How To Work With Video')
    window.show()
    sys.exit(app.exec())
