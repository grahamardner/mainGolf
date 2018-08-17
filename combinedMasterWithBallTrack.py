# Do in real time, and then save video file to play faster

import sys
import cv2
import imutils
import numpy as np
from collections import deque


from PyQt5.QtWidgets import QApplication, QMessageBox, QMainWindow, QFileDialog, QColorDialog
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi

from PyQt5 import QtCore
# from PyQt5.QtCore import pyqtSlot


class veryCoolMan(QMainWindow):

    def __init__(self):
        super(veryCoolMan, self).__init__()
        loadUi('masterComboUI.ui', self)

        self.startButton.clicked.connect(self.play_video)
        self.stopButton.clicked.connect(self.stop_video)
        self.nextButton.clicked.connect(self.next_frame)
        self.sendFrameButton.clicked.connect(self.sendImageToTab)

        self.trackButton.clicked.connect(self.track_ball_m2)

        self.actionOpen.triggered.connect(self.load_video)
        self.actionOpen.setShortcut("Ctrl+O")
        self.actionCloseVideo.triggered.connect(self.close_video)
        self.VideoImage = None
        self.boolIsLoaded = False
        self.blankImage = None
        self.sendImageHolder = None

        if self.boolIsLoaded is False:
            self.startButton.setEnabled(False)
            self.nextButton.setEnabled(False)
            self.stopButton.setEnabled(False)
            self.trackButton.setEnabled(False)
            self.sendFrameButton.setEnabled(False)

        # the below are imports from the old golfballsliders.py file

        self.image = None
        self.processedImage = None
        self.imageIsLoaded = False

        self.saveButtonIM.clicked.connect(self.saveClicked)
        self.visionButtonIM.clicked.connect(self.visionClicked)

        # connect the sliders to update the values for HSV
        self.uHSlider.valueChanged.connect(self.maskDetect)
        self.uSSlider.valueChanged.connect(self.maskDetect)
        self.uVSlider.valueChanged.connect(self.maskDetect)
        self.lHSlider.valueChanged.connect(self.maskDetect)
        self.lSSlider.valueChanged.connect(self.maskDetect)
        self.lVSlider.valueChanged.connect(self.maskDetect)

        # connect video start / end frame sliders to change those values
        self.videoFrontSlider.valueChanged.connect(self.slider_frame_numbers)
        self.videoBackSlider.valueChanged.connect(self.slider_frame_numbers)

        # connect the text boxes to update the values for HSV
        self.uHValue.returnPressed.connect(self.updateMaskImage)
        self.uSValue.returnPressed.connect(self.updateMaskImage)
        self.uVValue.returnPressed.connect(self.updateMaskImage)
        self.lHValue.returnPressed.connect(self.updateMaskImage)
        self.lSValue.returnPressed.connect(self.updateMaskImage)
        self.lVValue.returnPressed.connect(self.updateMaskImage)

        self.ballAValue.returnPressed.connect(self.ballValueChanged)
        self.ballBValue.returnPressed.connect(self.ballValueChanged)
        self.ballCValue.returnPressed.connect(self.ballValueChanged)
        self.ballDValue.returnPressed.connect(self.ballValueChanged)
        self.ballEValue.returnPressed.connect(self.ballValueChanged)
        self.ballFValue.returnPressed.connect(self.ballValueChanged)

        self.ballASlider.valueChanged.connect(self.ballDetect)
        self.ballBSlider.valueChanged.connect(self.ballDetect)
        self.ballCSlider.valueChanged.connect(self.ballDetect)
        self.ballDSlider.valueChanged.connect(self.ballDetect)
        self.ballESlider.valueChanged.connect(self.ballDetect)
        self.ballFSlider.valueChanged.connect(self.ballDetect)

        self.startBallButtonIM.clicked.connect(self.startBallClicked)
        self.findButtonIM.clicked.connect(self.ballDetectAlgo)

        self.widthValue.returnPressed.connect(self.loadImage)
        self.heightValue.returnPressed.connect(self.loadImage)
        self.colorButtonIM.clicked.connect(self.openColorDialog)

        self.chkDilate.stateChanged.connect(self.dilateMaskImage)
        self.chkClose.stateChanged.connect(self.dilateMaskImage)

    def close_video(self):  # TFIB (this function is broken)
        self.capture = None
        self.display_image(self.blankImage, 1)

    def load_video(self):  # loads video file and sets up the environment

        fname, filter = QFileDialog.getOpenFileName(self, 'Open Video File',
                                                    'Z:\Programming Learning\Python\Golf',
                                                    "Video Files (*.mpeg, *.avi)")
        # Makes a global var to hold the video file
        self.capture = cv2.VideoCapture(fname)

        # pretty self explanatory
        intFrameCount = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.frameNumberLabel.setText(str(intFrameCount))

        # sets up the sliders' min and max values relative to how many frames the video contains
        self.videoFrontSlider.setMinimum(0)
        self.videoFrontSlider.setMaximum(intFrameCount)
        self.videoBackSlider.setMinimum(0)
        self.videoBackSlider.setMaximum(intFrameCount)
        self.videoPositionSlider.setMinimum(0)
        self.videoPositionSlider.setMaximum(intFrameCount)

        # sets the initial start and end point to be the start and end of the video
        self.startFrameNum = 0
        self.endFrameNum = intFrameCount

        # sets the resolution for the imported video.  This should be adjusted to flex with
        # each video's particular resolution
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

        self.boolIsLoaded = True

    def slider_frame_numbers(self):
        self.startFrameNum = self.videoFrontSlider.value()
        self.endFrameNum = self.videoBackSlider.value()

    def play_video(self):
        # sets resolution for imported video.
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.startFrameNum)
        self.update_frame()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000.0/30)

    def next_frame(self):
        ret, self.VideoImage = self.capture.read()

        if self.VideoImage is None:
            self.stop_video()
        else:
            self.display_image(self.VideoImage, 1)

    def update_frame(self):
        intCurrentFrame = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
        if intCurrentFrame >= self.endFrameNum:
            self.stop_video()
        else:
            ret, self.VideoImage = self.capture.read()
            # copies the current frame so it can be sent to image tab if desired
            if self.VideoImage is None:
                print('got to stop video')
                self.stop_video()
            else:
                self.videoPositionSlider.setValue(intCurrentFrame+1)
                self.currentImage = self.VideoImage.copy()
                self.display_image(self.VideoImage, 1)

    def stop_video(self):
        self.timer.stop()

    def track_ball_m1(self):

        pts = deque(maxlen=64)
        greenLower = (0, 0, 39)
        greenUpper = (359, 65, 254)
        vs = self.capture
        # self.imgLabelVideo.clear()

        while True:

            frame = vs.read()
            # frame = self.capture.read()

            frame = frame[1]

            if frame is None:
                break

            frame = imutils.resize(frame, width=800)
            blurred = cv2.GaussianBlur(frame, (11, 11), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv, greenLower, greenUpper)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            # find contours in the mask and initialize the current
            # (x, y) center of the ball
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if imutils.is_cv2() else cnts[1]
            center = None

            # jprint(cnts)

            # only proceed if at least one contour was found
            if len(cnts) > 0:
                # find the largest contour in the mask, then use
                # it to compute the minimum enclosing circle and
                # centroid
                c = min(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size

                # if radius < 500:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

            # update the points queue
            pts.appendleft(center)

            # loop over the set of tracked points
            for i in range(1, len(pts)):
                # if either of the tracked points are None, ignore
                # them
                if pts[i - 1] is None or pts[i] is None:
                    continue

                # otherwise, compute the thickness of the line and
                # draw the connecting lines
                thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
                cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

            # show the frame to our screen
            self.display_image(frame, 1)

            key = cv2.waitKey(34) & 0xFF

            if key == ord("q"):
                break

    def track_ball_m2(self):  # this function implements simpleBlob to track ball
        self.slider_frame_numbers()
        pts = deque(maxlen=25)
        greenLower = (0, 0, 39)
        greenUpper = (359, 65, 254)
        vs = self.capture

        while True:

            frame = vs.read()
            frame = frame[1]

            if frame is None:
                break

            frame = imutils.resize(frame, width=1200)
            blurred = cv2.GaussianBlur(frame, (11, 11), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv, greenLower, greenUpper)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            # flip black and white
            mask = cv2.bitwise_not(mask)

            # Setup SimpleBlobDetector parameters.
            params = cv2.SimpleBlobDetector_Params()

            # Change thresholds
            params.minThreshold = 10
            params.maxThreshold = 150

            # Filter by Area.
            params.filterByArea = True
            params.minArea = 120

            # Filter by Circularity
            params.filterByCircularity = True
            params.minCircularity = 0.6

            # Filter by Convexity
            # params.filterByConvexity = True
            # params.minConvexity = 0.87

            # Filter by Inertia
            # params.filterByInertia = True
            # params.minInertiaRatio = 0.01

            # Create a detector with the parameters
            detector = cv2.SimpleBlobDetector_create(params)

            # Detect blobs.
            keypoints = detector.detect(mask)
            # print(keypoints)

            x = keypoints[0].pt[0]
            y = keypoints[0].pt[1]
            s = keypoints[0].size  # diameter
            # print(x)
            # print(y)
            # print(s)
            pts.appendleft((int(x), int(y)))
            # loop over the set of tracked points
            for i in range(1, len(pts)):
                # if either of the tracked points are None, ignore
                # them
                if pts[i - 1] is None or pts[i] is None:
                    continue

                # otherwise, compute the thickness of the line and
                # draw the connecting lines
                thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
                cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

            # im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array(
            #     []), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # cv2.circle(frame, (int(x), int(y)), 2, (255, 255, 0), -1)

            # cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
            cv2.putText(frame, 'Ball 1', (int(x)+10, int(y)-10),
                        cv2.FONT_HERSHEY_SIMPLEX,  .5, (50, 255, 50), 2)
            # show the frame to our screen
            self.display_image(frame, 1)

            key = cv2.waitKey(34) & 0xFF

            if key == ord("q"):
                break

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
        self.sendImageHolder = outImage
        if window == 1:
            self.imgLabelVideo.setPixmap(QPixmap.fromImage(outImage))
            self.imgLabelVideo.setScaledContents(True)

    def ballValueChanged(self):  # FCN gets changes in Ball Param Vals and calls ballDetect

        if int(self.ballAValue.text()) > 0 and int(self.ballAValue.text()) < 20:
            self.ballASlider.setValue(int(self.ballAValue.text()))
        else:
            QMessageBox.information(self, 'Error', 'Please Enter A Good Value')

        if int(self.ballBValue.text()) > 0 and int(self.ballBValue.text()) < 1000:
            self.ballBSlider.setValue(int(self.ballBValue.text()))
        else:
            QMessageBox.information(self, 'Error', 'Please Enter A Good Value')

        if int(self.ballCValue.text()) > 0 and int(self.ballCValue.text()) < 360:
            self.ballCSlider.setValue(int(self.ballCValue.text()))
        else:
            QMessageBox.information(self, 'Error', 'Please Enter A Good Value')

        if int(self.ballDValue.text()) > 0 and int(self.ballDValue.text()) < 360:
            self.ballDSlider.setValue(int(self.ballDValue.text()))
        else:
            QMessageBox.information(self, 'Error', 'Please Enter A Good Value')

        if int(self.ballEValue.text()) > 0 and int(self.ballEValue.text()) < 360:
            self.ballESlider.setValue(int(self.ballEValue.text()))
        else:
            QMessageBox.information(self, 'Error', 'Please Enter A Good Value')

        if int(self.ballFValue.text()) > 0 and int(self.ballFValue.text()) < 360:
            self.ballFSlider.setValue(int(self.ballFValue.text()))
        else:
            QMessageBox.information(self, 'Error', 'Please Enter A Good Value')

        self.loadButton.setDefault(False)
        self.loadButton.setAutoDefault(False)
        # self.ballDetect()

    def startBallClicked(self):  # Gets the ball detection ready upon a click on the "start ball decect" butto
        if self.imageIsLoaded is True:
            self.ballASlider.setValue(2)
            self.ballBSlider.setValue(140)
            self.ballCSlider.setValue(50)
            self.ballDSlider.setValue(10)
            self.ballESlider.setValue(0)
            self.ballFSlider.setValue(100)
            self.ballDetect()
        else:
            QMessageBox.information(self, 'Error', 'No image loaded')

    def ballDetect(self):  # this func updates text boxes nest to ball detect slides on image tab

        self.ballAValue.setText(str(self.ballASlider.value()))
        self.ballBValue.setText(str(self.ballBSlider.value()))
        self.ballCValue.setText(str(self.ballCSlider.value()))
        self.ballDValue.setText(str(self.ballDSlider.value()))
        self.ballEValue.setText(str(self.ballESlider.value()))
        self.ballFValue.setText(str(self.ballFSlider.value()))

        self.newCircleImage = self.image.copy()

    def ballDetectAlgo(self):
        A = 0
        B = 0
        C = 0
        D = 0
        E = 0
        F = 0

        A = int(self.ballASlider.value())
        B = int(self.ballBSlider.value())
        C = int(self.ballCSlider.value())
        D = int(self.ballDSlider.value())
        E = int(self.ballESlider.value())
        F = int(self.ballFSlider.value())

        maskBlur = cv2.medianBlur(self.justMaskedImage.copy(), 5)

        circles = cv2.HoughCircles(maskBlur, cv2.HOUGH_GRADIENT, A, B,
                                   param1=C, param2=D, minRadius=E, maxRadius=F)

        # if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(self.newCircleImage, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(self.newCircleImage, (i[0], i[1]), 2, (0, 0, 255), 3)

        self.processedImage = self.newCircleImage
        self.displayImage(2)
        # else:
        #    QMessageBox.information(self, 'Error', 'No Circles Found')

    def openColorDialog(self):
        colorMAN = QColorDialog.getColor()
        if self.chkLower.isChecked():
            self.labelColor.setStyleSheet(
                "QLabel#labelColor {background-color: %s}" % colorMAN.name())
            self.colorNameValue.setText(str(colorMAN))
        if self.chkUpper.isChecked():
            self.labelColorUpper.setStyleSheet(
                "QLabel#labelColorUpper {background-color: %s}" % colorMAN.name())
            self.colorNameValue.setText(str(colorMAN.name()))

    def updateImage(self):
        angle = int(self.rotateValue.text())
        self.loadButton.setDefault(False)
        self.loadButton.setAutoDefault(False)
        if angle >= 0 and angle <= 360:
            self.rotate_image(angle)
            self.dialValue.setValue(angle)
        else:
            QMessageBox.information(self, 'Error',
                                    'Please Enter Between 0 - 360')

    def updateMaskImage(self):

        if int(self.uHValue.text()) > 0 and int(self.uHValue.text()) < 360:
            self.uHSlider.setValue(int(self.uHValue.text()))
        else:
            QMessageBox.information(self, 'Error', 'Please Enter A Good Value')

        if int(self.uSValue.text()) > 0 and int(self.uSValue.text()) < 255:
            self.uSSlider.setValue(int(self.uSValue.text()))
        else:
            QMessageBox.information(self, 'Error', 'Please Enter A Good Value')

        if int(self.uVValue.text()) > 0 and int(self.uVValue.text()) < 255:
            self.uVSlider.setValue(int(self.uVValue.text()))
        else:
            QMessageBox.information(self, 'Error', 'Please Enter A Good Value')

        if int(self.lHValue.text()) > 0 and int(self.lHValue.text()) < 360:
            self.lHSlider.setValue(int(self.lHValue.text()))
        else:
            QMessageBox.information(self, 'Error', 'Please Enter A Good Value')

        if int(self.lSValue.text()) > 0 and int(self.lSValue.text()) < 255:
            self.lSSlider.setValue(int(self.lSValue.text()))
        else:
            QMessageBox.information(self, 'Error', 'Please Enter A Good Value')

        if int(self.lVValue.text()) > 0 and int(self.lVValue.text()) < 255:
            self.lVSlider.setValue(int(self.lVValue.text()))
        else:
            QMessageBox.information(self, 'Error', 'Please Enter A Good Value')

        self.loadButton.setDefault(False)
        self.loadButton.setAutoDefault(False)

        self.maskDetect()

    def dilateMaskImage(self):
        if self.chkDilate.isChecked():
            # gets value as an int from the text field in the GUI
            kernelInt = int(self.kernelSizeValue.text())

            # makes a kernel matrix frot the dilation function
            kernelSize = np.ones((kernelInt, kernelInt), np.uint8)

            ITERS = int(self.iterValue.text())

            dilation = cv2.dilate(self.processedImage, kernelSize, iterations=ITERS)
            self.processedImage = dilation
            self.justMaskedImage = self.processedImage.copy()
            self.displayImage(2)

        if self.chkClose.isChecked():
            kernelInt = int(self.kernelSizeValue.text())
            kernelSize = np.ones((kernelInt, kernelInt), np.uint8)
            closing = cv2.morphologyEx(self.processedImage, cv2.MORPH_CLOSE, kernelSize)
            self.processedImage = closing
            self.justMaskedImage = self.processedImage.copy()
            self.displayImage(2)

    def maskDetect(self):

        # "hsv" is an image file local to this function that saves the input
        # image as a file in the HSV color format
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # these two arrays hold the upper and lower color bounds for the HSV mask
        lowerHSVVals = np.array(
            [self.lHSlider.value(), self.lSSlider.value(), self.lVSlider.value()])
        upperHSVVals = np.array(
            [self.uHSlider.value(), self.uSSlider.value(), self.uVSlider.value()])
        self.uHValue.setText(str(self.uHSlider.value()))
        self.uSValue.setText(str(self.uSSlider.value()))
        self.uVValue.setText(str(self.uVSlider.value()))
        self.lHValue.setText(str(self.lHSlider.value()))
        self.lSValue.setText(str(self.lSSlider.value()))
        self.lVValue.setText(str(self.lVSlider.value()))

        # make an image var called "mask" that takes the masked image for later use
        mask = cv2.inRange(hsv, lowerHSVVals, upperHSVVals)

        self.globalLowHSV = lowerHSVVals
        self.globalHighHSV = upperHSVVals

        self.processedImage = mask
        self.justMaskedImage = self.processedImage.copy()
        self.displayImage(2)

    def visionClicked(self):
        if self.imageIsLoaded is True:
            # gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY) if len(
            #    self.image.shape) >= 3 else self.image

            # OLD lower_red = np.array([50, 0, 100])
            # OLD upper_red = np.array([255, 75, 255])

            # sets some semi-random default values for HSV upper and lower

            self.uHSlider.setValue(255)
            self.uSSlider.setValue(75)
            self.uVSlider.setValue(255)
            self.lHSlider.setValue(50)
            self.lSSlider.setValue(0)
            self.lVSlider.setValue(100)
            self.maskDetect()
        else:
            QMessageBox.information(self, 'Error', 'No image loaded')

    def sendImageToTab(self):
        self.image = self.sendImageHolder.copy()
        self.imageIsLoaded = True
        self.maskDetect()

    def loadClicked(self):
        # self.loadImage('small.PNG')
        fname, filter = QFileDialog.getOpenFileName(self, 'Open File',
                                                    'Z:\Programming Learning\Python\Golf\PyQtAtomTEst',
                                                    "Image Files (*.png, *.jpg)")
        if fname:
            self.loadImage(fname)
            self.imageIsLoaded = True
        else:
            print('Invalid Image')

    def saveClicked(self):
        fname, filter = QFileDialog.getSaveFileName(
            self, 'Save File', 'Z:\Programming Learning\Python\Golf\PyQtAtomTEst')
        if fname:
            cv2.imwrite(fname, self.processedImage)
        else:
            print('Error')

    def loadImage(self, fname):

        # Lots of the work happens in this function.  For each type of openCV
        # filter, you need to input the image in various forms, i.e. grayscale,
        # HSV mapped, RBG mapped, etc.  Depending on what the user selects,
        # appropriate copies are made of the imported photo with the correct
        # colar mapping applied so it can be utilized in the cannyDisplay fcn

        # maybe ... :)

        self.image = cv2.imread(fname, cv2.IMREAD_COLOR)
        if self.chkResize.isChecked():
            width = int(self.widthValue.text())
            height = int(self.heightValue.text())
            self.image = cv2.resize(self.image, (width, height))

        self.processedImage = self.image.copy()
        self.displayImage(1)

    def displayImage(self, window=1):
        qformat = QImage.Format_Indexed8

        if len(self.processedImage.shape) == 3:  # rows (0)cols (1),channels(2)
            if(self.processedImage.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.processedImage, self.processedImage.shape[1],
                     self.processedImage.shape[0],
                     self.processedImage.strides[0], qformat)

        # BGR --> RGB
        img = img.rgbSwapped()
        if window == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(img))
            self.imgLabel.setAlignment(QtCore.Qt.AlignHCenter |
                                       QtCore.Qt.AlignVCenter)
        if window == 2:
            self.imgLabel2.setPixmap(QPixmap.fromImage(img))
            self.imgLabel2.setAlignment(QtCore.Qt.AlignHCenter |
                                        QtCore.Qt.AlignVCenter)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = veryCoolMan()
    window.setWindowTitle('Learning How To Work With Video')
    window.show()
    sys.exit(app.exec())
