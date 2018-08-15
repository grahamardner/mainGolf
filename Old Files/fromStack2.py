import sys
# from PyQt5.QtWidgets import (QApplication, QAction, QMainWindow, QWidget, QPushButton, QFileDialog,
#                              QLabel, QFormLayout, QMessageBox)

from PyQt5.QtWidgets import QApplication, QAction, QMainWindow, QWidget, QPushButton, QFileDialog
from PyQt5.QtWidgets import QLabel, QFormLayout, QMessageBox

from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

import cv2


class VideoCaptureFunc(QWidget):
    def __init__(self, filename, parent):
        # could be:
        # super(QWidget, self).__init__()
        super(VideoCaptureFunc, self).__init__()

        self.cap = cv2.VideoCapture(str(filename))
        self.video_frame = QLabel()

        parent.layout.addWidget(self.video_frame)
        # lay = QVBoxLayout()
        # lay.setMargin(0)
        # lay.addWidget(self.video_frame)
        # self.setLayout(lay)

    def nextFrameSlot(self):
        ret, frame = self.cap.read()
        # My webcam yields frames in BGR format
        frame = cv2.cvtColor(frame, cv2.CV_BGR2RGB)
        img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        self.video_frame.setPixmap(pix)

    def start(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(1000./60)

    def pause(self):
        self.timer.stop()

    def deleteLater(self):
        self.cap.release()
        super(QWidget, self).deleteLater()


class VideoDisplayWidget(QWidget):
    def __init__(self, parent):
        super(VideoDisplayWidget, self).__init__(parent)

        self.layout = QFormLayout(self)

        self.startButton = QPushButton('Start', parent)
        self.startButton.clicked.connect(parent.startCapture)
        self.startButton.setFixedWidth(50)
        self.pauseButton = QPushButton('Pause', parent)
        self.pauseButton.setFixedWidth(50)
        self.layout.addRow(self.startButton, self.pauseButton)
#       self.setLayout(self.layout)


class ControlWindow(QMainWindow):
    def __init__(self):
        super(ControlWindow, self).__init__()
        self.setGeometry(50, 50, 800, 600)
        # self.setGeometry(50, 50, 1400, 900)
        self.setWindowTitle("PyTrack")

        self.capture = None

        self.videoFileName = None

        self.isVideoFileLoaded = False

        self.quitAction = QAction("&Exit", self)
        self.quitAction.setShortcut("Ctrl+Q")
        self.quitAction.setStatusTip('Close The App')
        self.quitAction.triggered.connect(self.closeApplication)

        self.openVideoFile = QAction("&Open Video File", self)
        self.openVideoFile.setShortcut("Ctrl+Shift+V")
        self.openVideoFile.setStatusTip('Open .h264 File')
        self.openVideoFile.triggered.connect(self.loadVideoFile)

        self.mainMenu = self.menuBar()
        self.fileMenu = self.mainMenu.addMenu('&File')
        self.fileMenu.addAction(self.openVideoFile)
        self.fileMenu.addAction(self.quitAction)

        self.videoDisplayWidget = QWidget(self)
        self.setCentralWidget(self.videoDisplayWidget)

        # self.imageCaptureWindow = QWidget(self)
        # self.start_button = QPushButton('Start', self.imageCaptureWindow)
        # self.start_button.clicked.connect(self.startCapture)
        # self.start_button.setGeometry(0, 10, 40, 30)
        # self.pause_button = QPushButton('Pause', self.imageCaptureWindow)
        # self.pause_button.setGeometry(50, 10, 40, 30)

    def startCapture(self):
        if not self.capture and self.isPositionFileLoaded and self.isVideoFileLoaded:
            self.capture = VideoCaptureFunc(self.videoFileName, self.videoDisplayWidget)
            self.videoDisplayWidget.pauseButton.clicked.connect(self.capture.pause)
        self.capture.start()

    def endCapture(self):
        self.capture.deleteLater()
        self.capture = None

    def loadVideoFile(self):
        # try:
        self.videoFileName = QFileDialog.getOpenFileName(self, 'Select .h264 Video File')
        self.isVideoFileLoaded = True
        # except ValueError:
        #    print("Please select a .h264 file")

    def closeApplication(self):
        choice = QMessageBox.question(self, 'Message', 'Do you really want to exit?',
                                            QMessageBox.Yes | QMessageBox.No)
        if choice == QMessageBox.Yes:
            print("Closing....")
            sys.exit()
        else:
            pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ControlWindow()
    sys.exit(app.exec_())
