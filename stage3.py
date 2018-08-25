# Do in real time, and then save video file to play faster

import sys
import cv2
import imutils
import numpy as np
import qdarkstyle
from collections import deque


from PyQt5.QtWidgets import QApplication, QMessageBox, QMainWindow, QFileDialog, QColorDialog
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi

from PyQt5 import QtCore


class veryCoolMan(QMainWindow):

    def __init__(self):
        super(veryCoolMan, self).__init__()
        loadUi('stage3.ui', self)

        self.startButton.clicked.connect(self.VT_play_video)
        self.stopButton.clicked.connect(self.VT_stop_video)
        self.nextButton.clicked.connect(self.VT_next_frame)
        self.sendFrameButton.clicked.connect(self.VT_send_image_to_tab)

        self.actionOpenVideo.triggered.connect(self.MB_load_video)
        self.actionOpenVideo.setShortcut("Ctrl+O")
        self.actionCloseVideo.triggered.connect(self.MB_close_video)
        self.actionOpenImage.triggered.connect(self.MB_load_image)

        self.VideoImage = None
        self.blankImage = None
        self.sendImageHolder = None

        # This variable will hold the current image displayed in video and send it over
        self.globalImage = None

        # the below are imports from the old golfballsliders.py file

        self.image = None
        self.processedImage = None
        self.imgCropped = None

        # bools to see if things have been loaded
        self.boolImageIsLoaded = False
        self.boolVideoIsLoaded = False
        self.boolBallTrackingPermitted = False
        self.boolNeedsBWFlip = True
        self.boolValuesCommitted = False

        # Bools to keep the sliders from updating when we are initializing
        self.boolMaskSlidersAdjust = False
        self.boolBallDetectSliderAdjust = False

        # initialize the radio buttons
        self.radCrop.setChecked(False)
        self.radFull.setChecked(True)
        self.radCrop.clicked.connect(self.IT_radio_toggle)
        self.radFull.clicked.connect(self.IT_radio_toggle)

        # connect video start / end frame sliders to change those values
        self.videoFrontSlider.valueChanged.connect(self.VT_slider_frame_numbers)
        self.videoBackSlider.valueChanged.connect(self.VT_slider_frame_numbers)

        # connect the text boxes to update the values for HSV
        self.uHValue.returnPressed.connect(self.update_mask_slider_values)
        self.uSValue.returnPressed.connect(self.update_mask_slider_values)
        self.uVValue.returnPressed.connect(self.update_mask_slider_values)
        self.lHValue.returnPressed.connect(self.update_mask_slider_values)
        self.lSValue.returnPressed.connect(self.update_mask_slider_values)
        self.lVValue.returnPressed.connect(self.update_mask_slider_values)

        # connect the sliders to update the values for HSV
        self.uHSlider.valueChanged.connect(self.update_mask_textbox_values)
        self.uSSlider.valueChanged.connect(self.update_mask_textbox_values)
        self.uVSlider.valueChanged.connect(self.update_mask_textbox_values)
        self.lHSlider.valueChanged.connect(self.update_mask_textbox_values)
        self.lSSlider.valueChanged.connect(self.update_mask_textbox_values)
        self.lVSlider.valueChanged.connect(self.update_mask_textbox_values)

        self.valueThreshMinIT.returnPressed.connect(self.update_ball_detect_slider_values)
        self.valueThreshMaxIT.returnPressed.connect(self.update_ball_detect_slider_values)
        self.valueFilterByAreaIT.returnPressed.connect(self.update_ball_detect_slider_values)
        self.valueFilterByAreaMaxIT.returnPressed.connect(self.update_ball_detect_slider_values)
        self.valueFilterByCircIT.returnPressed.connect(self.update_ball_detect_slider_values)

        self.sliderThreshMinIT.valueChanged.connect(self.update_ball_detect_textbox_values)
        self.sliderThreshMaxIT.valueChanged.connect(self.update_ball_detect_textbox_values)
        self.sliderFilterByAreaIT.valueChanged.connect(self.update_ball_detect_textbox_values)
        self.sliderFilterByAreaMaxIT.valueChanged.connect(self.update_ball_detect_textbox_values)
        self.sliderFilterByCircIT.valueChanged.connect(self.update_ball_detect_textbox_values)

        self.colorButtonIM.clicked.connect(self.IT_open_color_dialog)

        self.btnCommitParamsIT.clicked.connect(self.IT_check_commit_params)
        self.btnSelectROIIT.clicked.connect(self.IT_IP_roi)
        self.btnCropActionIT.clicked.connect(self.IT_IP_swap)
        self.btnResetImgIT.clicked.connect(self.IT_IP_reset_image)

        self.chkDilate.stateChanged.connect(self.IT_IP_morphology)
        self.chkClose.stateChanged.connect(self.IT_IP_morphology)

        self.chkBallDetect.stateChanged.connect(self.IT_enab_ball_detect)
        self.chkMorphVT.stateChanged.connect(self.IT_enab_morph)
        self.chkInRangeMasking.stateChanged.connect(self.IT_enab_masking)

        # now, whenever we change tabs, we double check to make sure the correct
        # buttons are grayed out / not grated out
        self.tabWidget.currentChanged.connect(self.VT_update_button_status)

        # checks the state of the program and disables buttons as needed
        self.VT_update_button_status()
        self.IT_update_button_status()

        # Sets startup tab to video tab
        self.tabWidget.setCurrentIndex(0)

    def IT_radio_toggle(self):
        print('got to the toggle! GET TO THE CHOPPAHH!')
        if self.radFull.isChecked():
            print('slid into the DMs')
            self.radCrop.setChecked(True)
            self.radFull.setChecked(False)
        if self.radCrop.isChecked():
            print('slid off the slip and slide')
            self.radCrop.setChecked(False)
            self.radFull.setChecked(True)

    def IT_check_commit_params(self):

        if self.chkHSVCommitIT.isChecked():
            if self.chkMorphCommitIT.isChecked():
                if self.chkBallDetectCommitIT.isChecked():
                    self.VT_ball_params()
                    self.boolBallTrackingPermitted = True
                    self.VT_update_button_status()

        else:
            QMessageBox.information(self, 'Error', 'Not All Parameters Correct')

    def VT_ball_params(self):

        # Jump over to video tab
        self.tabWidget.setCurrentIndex(0)

        # Update all text labels to show the current settings

        if self.chkInRangeMasking.isChecked():
            self.lblHSVTrueVT.setText('True')

            self.lblUH.setText(str(self.uHValue.text()))
            self.lblUS.setText(str(self.uSValue.text()))
            self.lblUV.setText(str(self.uVValue.text()))
            self.lblLH.setText(str(self.lHValue.text()))
            self.lblLS.setText(str(self.lSValue.text()))
            self.lblLV.setText(str(self.lVValue.text()))
        else:
            self.lblHSVTrueVT.setText('False')

        if self.chkMorphVT.isChecked():  # Check for all  morphology values
            self.lblMorphTrueVT.setText('True')

            if self.chkClose.isChecked():
                self.lblCloseTrueVT.setText('True')
            else:
                self.lblCloseTrueVT.setText('False')

            if self.chkDilate.isChecked():
                self.lblDilateTrueVT.setText('True')
            else:
                self.lblDilateTrueVT.setText('False')

            self.lblKernelSizeVT.setText(str(self.kernelSizeValue.text()))
            self.lblIterVT.setText(str(self.iterValue.text()))
        else:
            self.lblMorphTrueVT.setText('False')

        if self.chkBallDetect.isChecked():
            self.lblBallTrackTrueVT.setText('True')

            if self.chkEnableThreshIT.isChecked():
                self.lblThreshTrueVT.setText('True')
                self.lblThreshMinVT.setText(str(self.valueThreshMinIT.text()))
                self.lblThreshMaxVT.setText(str(self.valueThreshMaxIT.text()))

            else:
                self.lblThreshTrueVT.setText('False')

            if self.chkFilterByAreaIT.isChecked():
                self.lblAreaTrueVT.setText('True')
                self.lblMinAreaVT.setText(str(self.valueFilterByAreaIT.text()))
            else:
                self.lblAreaTrueVT.setText('False')

            if self.chkFilterByCircIT.isChecked():
                self.lblCircTrueVT.setText('True')
                self.lblMinCircVT.setText(str(self.valueFilterByCircIT.text()))
            else:
                self.lblCircTrueVT.setText('False')
        else:
            self.lblBallTrackTrueVT.setText('False')

    def VT_update_button_status(self):  # checks the state of the program and disables buttons as needed

        if self.boolVideoIsLoaded is True:
            self.startButton.setEnabled(True)
            self.nextButton.setEnabled(True)
            self.stopButton.setEnabled(True)
            self.sendFrameButton.setEnabled(True)

        if self.boolVideoIsLoaded is False:
            self.startButton.setEnabled(False)
            self.nextButton.setEnabled(False)
            self.stopButton.setEnabled(False)
            self.sendFrameButton.setEnabled(False)

        if self.boolBallTrackingPermitted is False:
            self.chkTrackBall.setEnabled(False)

        if self.boolBallTrackingPermitted is True:
            self.chkTrackBall.setEnabled(True)

    def VT_slider_frame_numbers(self):  # Updates start and stop points of video playback

        self.startFrameNum = self.videoFrontSlider.value()
        self.endFrameNum = self.videoBackSlider.value()  # gets the start and stop positions (as frame #'s)

    def VT_play_video(self):  # plays the video from the starting frame # continuously

        # sets resolution for imported video.
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.startFrameNum)

        if self.chkTrackBall.isChecked():  # play video with ball tracking enabled

            # make sure you set up all the stuff you need in the image tab

            if self.params is None:
                QMessageBox.information(
                    self, 'Error', 'Blob Detect parameters have not been set in Image Tab Yet')

            else:
                self.pts = deque(maxlen=25)
                self.VT_update_frame_tracking()
                self.timer = QTimer(self)
                self.timer.timeout.connect(self.VT_update_frame_tracking)
                self.timer.start(1000.0/30)

        else:  # play video without ball tracking

            self.VT_update_frame()
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.VT_update_frame)
            self.timer.start(1000.0/30)

    def VT_next_frame(self):  # shows only the next frame of the video
        ret, self.VideoImage = self.capture.read()

        if self.VideoImage is None:
            self.VT_stop_video()
        else:
            self.VT_display_video_frame(self.VideoImage, 1)

    def VT_update_frame(self):  # shows the next frame in the window without tracking
        intCurrentFrame = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
        if intCurrentFrame >= self.endFrameNum:
            self.VT_stop_video()
        else:
            ret, self.VideoImage = self.capture.read()
            # copies the current frame so it can be sent to image tab if desired
            if self.VideoImage is None:
                print('got to stop video')
                self.VT_stop_video()
            else:
                self.videoPositionSlider.setValue(intCurrentFrame+1)
                self.globalImage = self.VideoImage.copy()
                self.VT_display_video_frame(self.VideoImage, 1)

    def VT_update_frame_tracking(self):  # shows the next frame in the window with tracking
        intCurrentFrame = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
        if intCurrentFrame >= self.endFrameNum:
            self.VT_stop_video()
        else:
            ret, self.VideoImage = self.capture.read()
            # copies the current frame so it can be sent to image tab if desired
            if self.VideoImage is None:
                print('got to stop video')
                self.VT_stop_video()
            else:
                frame = self.VideoImage
                # frame = frame[1]

                frame = imutils.resize(frame, width=600)
                blurred = cv2.GaussianBlur(frame, (11, 11), 0)
                hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

                mask = cv2.inRange(hsv, self.globalLowHSV, self.globalHighHSV)
                mask = cv2.erode(mask, None, iterations=2)
                mask = cv2.dilate(mask, None, iterations=2)

                # flip black and white
                mask = cv2.bitwise_not(mask)

                # Create a detector with the parameters
                detector = cv2.SimpleBlobDetector_create(self.params)

                # Detect blobs.
                keypoints = detector.detect(mask)

                if len(keypoints) == 0:
                    QMessageBox.information(self, 'Error', 'No Balls Found')
                    self.chkTrackBall.setChecked(False)
                    # need to send back to frame 1 and pause
                    self.VT_stop_video()

                x = keypoints[0].pt[0]
                y = keypoints[0].pt[1]
                # s = keypoints[0].size  # diameter

                self.pts.appendleft((int(x), int(y)))
                # loop over the set of tracked points
                for i in range(1, len(self.pts)):
                    # if either of the tracked points are None, ignore
                    # them
                    if self.pts[i - 1] is None or self.pts[i] is None:
                        continue

                    # otherwise, compute the thickness of the line and
                    # draw the connecting lines
                    thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
                    cv2.line(frame, self.pts[i - 1], self.pts[i], (0, 0, 255), thickness)

                cv2.putText(frame, 'Ball 1', (int(x)+10, int(y)-10),
                            cv2.FONT_HERSHEY_SIMPLEX,  .5, (50, 255, 50), 2)

                self.videoPositionSlider.setValue(intCurrentFrame+1)
                self.VideoImage = frame
                self.globalImage = self.VideoImage.copy()
                self.VT_display_video_frame(self.VideoImage, 1)

    def VT_display_video_frame(self, img, window=1):  # paints the current frame into the label
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

    def VT_send_image_to_tab(self):  # Takes the current frame displayed and sends it to the image tab

        # the command below makes it so you can't load the image tab.
        # self.tabWidget.setTabEnabled(1, False)

        # The command below chagnes the tab to the Image tab (index 1)
        self.tabWidget.setCurrentIndex(1)

        self.boolImageIsLoaded = True
        self.IT_update_button_status()

        self.processedImage = self.globalImage.copy()
        self.IT_display_image(1)

    def VT_stop_video(self):  # stops video playback
        self.timer.stop()
        # self.display_image(self.currentImage, 1)

    def IT_update_button_status(self):   # Each time a checkbox is checked on the image tab
                                                # this function makes sure the right buttons get disabled
        if self.boolImageIsLoaded is False:
            self.toolBox.setEnabled(False)
            self.btnCommitParamsIT.setEnabled(False)

        if self.boolImageIsLoaded is True:
            self.toolBox.setEnabled(True)
            # We want to enable the overall tab widget but not enable all sliders
            self.chkClose.setEnabled(False)
            self.chkDilate.setEnabled(False)
            self.kernelSizeValue.setEnabled(False)
            self.iterValue.setEnabled(False)
            self.btnCommitParamsIT.setEnabled(True)

    def IT_IP_roi(self):
        self.roi = cv2.selectROI(self.processedImage)
        self.textBrowser.setText(str(self.roi))
        # Crop image
        self.imgCropped = self.processedImage[int(
            self.roi[1]):int(self.roi[1]+self.roi[3]), int(self.roi[0]):int(self.roi[0]+self.roi[2])]

    def IT_IP_swap(self):
        self.processedImage = self.imgCropped.copy()
        self.IT_display_image(1)

    def IT_IP_reset_image(self):
        self.processedImage = self.globalImage.copy()
        self.chkInRangeMasking.setChecked(False)
        self.chkMorphVT.setChecked(False)
        self.chkBallDetect.setChecked(False)
        self.IT_display_image(1)

    def IT_enab_masking(self):  # enables inRange masking in IT, sets default vals and calls mask
        if self.chkInRangeMasking.isChecked() is True:

            print('enabling all the crap')
            self.uHSlider.setEnabled(True)
            self.uSSlider.setEnabled(True)
            self.uVSlider.setEnabled(True)
            self.lHSlider.setEnabled(True)
            self.lSSlider.setEnabled(True)
            self.lVSlider.setEnabled(True)
            self.uHValue.setEnabled(True)
            self.uSValue.setEnabled(True)
            self.uVValue.setEnabled(True)
            self.lHValue.setEnabled(True)
            self.lSValue.setEnabled(True)
            self.lVValue.setEnabled(True)

            # Set up all the masking sliders and check boxes
            self.boolMaskSlidersAdjust = False
            self.uHSlider.setValue(359)
            self.uSSlider.setValue(65)
            self.uVSlider.setValue(254)
            self.lHSlider.setValue(0)
            self.lSSlider.setValue(0)
            self.lVSlider.setValue(39)

            # these two arrays hold the upper and lower color bounds for the HSV mask
            self.globalLowHSV = np.array(
                [self.lHSlider.value(), self.lSSlider.value(), self.lVSlider.value()])
            self.globalHighHSV = np.array(
                [self.uHSlider.value(), self.uSSlider.value(), self.uVSlider.value()])

            self.boolMaskSlidersAdjust = True

            if self.boolImageIsLoaded is True:
                self.IT_IP_inrange_mask()
        else:
            # Disable all the crap
            print('Disabling lots of crap')
            self.uHSlider.setEnabled(False)
            self.uSSlider.setEnabled(False)
            self.uVSlider.setEnabled(False)
            self.lHSlider.setEnabled(False)
            self.lSSlider.setEnabled(False)
            self.lVSlider.setEnabled(False)
            self.uHValue.setEnabled(False)
            self.uSValue.setEnabled(False)
            self.uVValue.setEnabled(False)
            self.lHValue.setEnabled(False)
            self.lSValue.setEnabled(False)
            self.lVValue.setEnabled(False)

    def IT_enab_morph(self):

        if self.chkMorphVT.isChecked():
            self.kernelSizeValue.setEnabled(True)
            self.iterValue.setEnabled(True)
            self.kernelSizeValue.setText('4')
            self.iterValue.setText('1')
            self.chkClose.setEnabled(True)
            self.chkDilate.setEnabled(True)

        else:
            self.kernelSizeValue.setText(' ')
            self.iterValue.setText(' ')
            self.chkClose.setEnabled(False)
            self.chkDilate.setEnabled(False)
            self.kernelSizeValue.setEnabled(False)
            self.iterValue.setEnabled(False)

    def IT_enab_ball_detect(self):  # enables ball detect in IT, sets default vals and calls ball detect

        if self.chkBallDetect.isChecked() is True:

            # Turn on all the crap
            self.chkEnableThreshIT.setEnabled(True)
            self.chkFilterByAreaIT.setEnabled(True)
            self.chkFilterByCircIT.setEnabled(True)
            self.sliderThreshMinIT.setEnabled(True)
            self.sliderThreshMaxIT.setEnabled(True)
            self.sliderFilterByAreaIT.setEnabled(True)
            self.sliderFilterByAreaMaxIT.setEnabled(True)
            self.sliderFilterByCircIT.setEnabled(True)

            # Stops the sliders adjusting so we can update values quickly
            self.boolBallDetectSliderAdjust = False

            # Checks the boxes for all three thingies automatically
            self.chkEnableThreshIT.setChecked(True)
            self.chkFilterByAreaIT.setChecked(True)
            self.chkFilterByCircIT.setChecked(True)

            self.sliderThreshMinIT.setValue(10)
            self.sliderThreshMaxIT.setValue(150)
            self.sliderFilterByAreaIT.setValue(120)
            self.sliderFilterByAreaMaxIT.setValue(500000)
            self.sliderFilterByCircIT.setValue(60)

            # Turns back on the auto update image after slider is moved
            self.boolBallDetectSliderAdjust = True

            # Call the actual ball detect function if an image is loaded
            if self.boolImageIsLoaded is True:
                self.IT_IP_ball_detect()
        else:
            # Stops the sliders adjusting so we can update values quickly
            self.boolBallDetectSliderAdjust = False

            self.chkEnableThreshIT.setChecked(False)
            self.chkFilterByAreaIT.setChecked(False)
            self.chkFilterByCircIT.setChecked(False)
            self.sliderThreshMinIT.setValue(1)
            self.sliderThreshMaxIT.setValue(1)
            self.sliderFilterByAreaIT.setValue(1)
            self.sliderFilterByAreaMaxIT.setValue(1)
            self.sliderFilterByCircIT.setValue(1)

            # Turns back on the auto update image after slider is moved
            self.boolBallDetectSliderAdjust = True

            # Turn off all the crap
            self.chkEnableThreshIT.setEnabled(False)
            self.chkFilterByAreaIT.setEnabled(False)
            self.chkFilterByCircIT.setEnabled(False)
            self.sliderThreshMinIT.setEnabled(False)
            self.sliderThreshMaxIT.setEnabled(False)
            self.sliderFilterByAreaIT.setEnabled(False)
            self.sliderFilterByAreaMaxIT.setEnabled(False)
            self.sliderFilterByCircIT.setEnabled(False)

    def IT_IP_inrange_mask(self):  # Applies the inRange color mask to the image in the Image Tab

        # "hsv" is an image file local to this function that saves the input
        # image as a file in the HSV color format
        hsv = cv2.cvtColor(self.processedImage, cv2.COLOR_BGR2HSV)

        # make an image var called "mask" that takes the masked image for later use
        mask = cv2.inRange(hsv, self.globalLowHSV, self.globalHighHSV)

        self.processedImage = mask
        self.IT_display_image(1)

    def IT_IP_morphology(self):  # Applies morphology if desired
        if self.chkDilate.isChecked():
            # gets value as an int from the text field in the GUI
            kernelInt = int(self.kernelSizeValue.text())

            # makes a kernel matrix frot the dilation function
            kernelSize = np.ones((kernelInt, kernelInt), np.uint8)

            ITERS = int(self.iterValue.text())

            dilation = cv2.dilate(self.processedImage, kernelSize, iterations=ITERS)
            self.processedImage = dilation
            self.justMaskedImage = self.processedImage.copy()
            self.IT_display_image(1)

        if self.chkClose.isChecked():
            kernelInt = int(self.kernelSizeValue.text())
            kernelSize = np.ones((kernelInt, kernelInt), np.uint8)
            closing = cv2.morphologyEx(self.processedImage, cv2.MORPH_CLOSE, kernelSize)
            self.processedImage = closing
            self.justMaskedImage = self.processedImage.copy()
            self.IT_display_image(1)

    def IT_IP_ball_detect(self):  # Does the processing work to show updated ball in image tab
        if self.boolImageIsLoaded is True:
            # Setup SimpleBlobDetector parameters.
            self.params = cv2.SimpleBlobDetector_Params()

            # Change thresholds
            self.params.minThreshold = self.sliderThreshMinIT.value()
            self.params.maxThreshold = self.sliderThreshMaxIT.value()

            # Filter by Area.
            if self.chkFilterByAreaIT.isChecked():
                self.params.filterByArea = True
                self.params.minArea = self.sliderFilterByAreaIT.value()
                self.params.maxArea = self.sliderFilterByAreaMaxIT.value()
            else:
                self.params.filterByArea = False

            # Filter by Circularity
            if self.chkFilterByAreaIT.isChecked():
                self.params.filterByCircularity = True
                self.params.minCircularity = 0.6
            else:
                self.params.filterByCircularity = False

            # Perform SimpleBlobDetector Ball Detection
            frame = self.processedImage
            blurred = cv2.GaussianBlur(frame, (11, 11), 0)

            # if self.boolNeedsBWFlip is True:
            mask = cv2.bitwise_not(blurred)
            #    self.boolNeedsBWFlip = False

            # Create a detector with the parameters
            detector = cv2.SimpleBlobDetector_create(self.params)
            if self.radFull.isChecked():
                holderHolder = self.globalImage
            else:
                if self.imgCropped is None:
                    QMessageBox.information(
                        self, 'Error', 'No Cropped Image Found.  Using Full Image.')
                    holderHolder = self.globalImage
                else:
                    holderHolder = self.imgCropped
            # Detect blobs.
            keypoints = detector.detect(mask)
            if len(keypoints) > 0:
                self.textBrowser.setText('At Least One Ball Was Found')
                x = keypoints[0].pt[0]
                y = keypoints[0].pt[1]
                # s = keypoints[0].size  # diameter
                # cv2.circle(holderHolder, (int(x), int(y)), 2, (0, 0, 255), -1)
                holderHolder = cv2.drawKeypoints(holderHolder, keypoints, np.array(
                    []), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.putText(holderHolder, 'Ball 1', (int(x)+10, int(y)-10),
                            cv2.FONT_HERSHEY_SIMPLEX,  .5, (50, 255, 50), 2)

            else:
                self.textBrowser.setText('No Balls Found')
            self.processedImage = holderHolder
            self.IT_display_image(1)

    def IT_open_color_dialog(self):  # Opens a color picker dialog box in Image Tab
        colorMAN = QColorDialog.getColor()
        if self.chkLower.isChecked():
            self.labelColor.setStyleSheet(
                "QLabel#labelColor {background-color: %s}" % colorMAN.name())
            self.colorNameValue.setText(str(colorMAN))
        if self.chkUpper.isChecked():
            self.labelColorUpper.setStyleSheet(
                "QLabel#labelColorUpper {background-color: %s}" % colorMAN.name())
            self.colorNameValue.setText(str(colorMAN.name()))

    def update_mask_slider_values(self):  # Updates mask slider values in IT

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

        # If an image is loaded, call the update mask function
        if self.boolImageIsLoaded is True:
            if self.boolMaskSlidersAdjust is True:
                self.IT_IP_inrange_mask()

    def update_mask_textbox_values(self):  # Updates mask textbox values in IT

        self.uHValue.setText(str(self.uHSlider.value()))
        self.uSValue.setText(str(self.uSSlider.value()))
        self.uVValue.setText(str(self.uVSlider.value()))
        self.lHValue.setText(str(self.lHSlider.value()))
        self.lSValue.setText(str(self.lSSlider.value()))
        self.lVValue.setText(str(self.lVSlider.value()))

        # if an image is loaded, call the update mask function
        if self.boolImageIsLoaded is True:
            if self.boolMaskSlidersAdjust is True:
                self.IT_IP_inrange_mask()
        else:
            print('no image loaded')

    def update_ball_detect_slider_values(self):  # Updates ball detect textbox values in IT

        if int(self.valueThreshMinIT.text()) > 0 and int(self.valueThreshMinIT.text()) < 20:
            self.sliderThreshMinIT.setValue(int(self.valueThreshMinIT.text()))
        else:
            QMessageBox.information(self, 'Error', 'Outside threshold min val')

        if int(self.valueThreshMaxIT.text()) > 0 and int(self.valueThreshMaxIT.text()) < 1001:
            self.sliderThreshMaxIT.setValue(int(self.valueThreshMaxIT.text()))
        else:
            QMessageBox.information(self, 'Error', 'Outside threshold max val')

        if int(self.valueFilterByAreaIT.text()) > 0 and int(self.valueFilterByAreaIT.text()) < 360:
            self.sliderFilterByAreaIT.setValue(int(self.valueFilterByAreaIT.text()))
        else:
            QMessageBox.information(self, 'Error', 'Outside threshold min area value')

        if int(self.valueFilterByAreaMaxIT.text()) > 0 and int(self.valueFilterByAreaMaxIT.text()) < 9000001:
            self.sliderFilterByAreaMaxIT.setValue(int(self.valueFilterByAreaMaxIT.text()))
        else:
            QMessageBox.information(self, 'Error', 'Outside max area value')

        if int(self.valueFilterByCircIT.text()) > 0 and int(self.valueFilterByCircIT.text()) < 101:
            self.sliderFilterByCircIT.setValue(int(self.valueFilterByCircIT.text()))
        else:
            QMessageBox.information(self, 'Error', 'Circularity must be between 0-100')

        if self.boolImageIsLoaded is True:
            if self.boolBallDetectSliderAdjust is True:
                self.IT_IP_ball_detect()

    def update_ball_detect_textbox_values(self):  # Updates ball detect text boxes in IT

        self.valueThreshMinIT.setText(str(self.sliderThreshMinIT.value()))
        self.valueThreshMaxIT.setText(str(self.sliderThreshMaxIT.value()))
        self.valueFilterByAreaIT.setText(str(self.sliderFilterByAreaIT.value()))
        self.valueFilterByCircIT.setText(str(self.sliderFilterByCircIT.value()))
        self.valueFilterByAreaMaxIT.setText(str(self.sliderFilterByAreaMaxIT.value()))

        if self.boolImageIsLoaded is True:
            if self.boolBallDetectSliderAdjust is True:
                self.IT_IP_ball_detect()

    def MB_load_image(self):  # Load an image to be processed in the image tab
        # self.loadImage('small.PNG')
        fname, filter = QFileDialog.getOpenFileName(self, 'Open File',
                                                    'Z:\Programming Learning\Python\Golf\PyQtAtomTEst',
                                                    "Image Files (*.png, *.jpg)")
        if fname:
            self.boolImageIsLoaded = True
            self.IT_update_button_status()
            self.processedImage = cv2.imread(fname, cv2.IMREAD_COLOR)
            self.globalImage = self.processedImage.copy()
            self.IT_display_image(1)
            # go over to the image tab to see the image you loaded
            self.tabWidget.setCurrentIndex(1)

            # This code allows you to resize.  Old version of GUI has buttons for this functionality

            # if self.chkResize.isChecked():
            #    width = int(self.widthValue.text())
            #    height = int(self.heightValue.text())
            #    self.image = cv2.resize(self.image, (width, height))

        else:
            print('Invalid Image')

    def MB_save_image(self):  # Save the current image displayed in the image tab

        # Check to make sure there is an image that can be saved
        if self.processedImage is None:
            QMessageBox.information(self, 'Error', 'No image to save')

        # now that we know there is an image, get a filename from a save dialog box
        # and write image to that directory
        else:
            fname, filter = QFileDialog.getSaveFileName(
                self, 'Save File', 'Z:\Programming Learning\Python\Golf\PyQtAtomTEst')
            if fname:
                cv2.imwrite(fname, self.processedImage)
            else:
                print('Error')

    def MB_close_video(self):  # TFIB (this function is broken)
        self.capture = None
        self.VT_display_video_frame(self.blankImage, 1)

    def MB_load_video(self):  # loads video file and sets up the environment

        # Get file name of video that you want to load
        fname, filter = QFileDialog.getOpenFileName(self, 'Open Video File',
                                                    'Z:\Programming Learning\Python\Golf',
                                                    "Video Files (*.mpeg, *.avi)")
        # Makes a global var to hold the video file
        self.capture = cv2.VideoCapture(fname)

        # Gets total # of frames in video and makes the number globally available
        intFrameCount = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.frameNumberLabel.setText(str(intFrameCount))

        # Gets FPS of video and makes the number globally available
        self.intFPS = self.capture.get(cv2.CAP_PROP_FPS)
        self.labelFPS.setText(str(self.intFPS))

        # Displays length of video
        intVideoLength = intFrameCount / self.intFPS
        self.labelVideoLengthSeconds.setText(str(intVideoLength))

        # sets up the sliders' min and max values relative to how many frames the video contains
        self.videoFrontSlider.setMinimum(0)
        self.videoFrontSlider.setMaximum(intFrameCount)
        self.videoBackSlider.setMinimum(0)
        self.videoBackSlider.setMaximum(intFrameCount)
        self.videoPositionSlider.setMinimum(0)
        self.videoPositionSlider.setMaximum(intFrameCount)
        self.videoBackSlider.setValue(intFrameCount)

        # sets the initial start and end point to be the start and end of the video
        self.startFrameNum = 0
        self.endFrameNum = intFrameCount

        # sets the resolution for the imported video.  This should be adjusted to flex with
        # each video's particular resolution
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

        self.boolVideoIsLoaded = True
        self.VT_update_button_status()

    def IT_display_image(self, window=1):  # Displays image in the Image Tab (not used for video)
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
            self.labelPictureDisplay.setPixmap(QPixmap.fromImage(img))
            self.labelPictureDisplay.setAlignment(QtCore.Qt.AlignHCenter |
                                                  QtCore.Qt.AlignVCenter)
        if window == 2:
            self.labelPictureDisplay.setPixmap(QPixmap.fromImage(img))
            self.labelPictureDisplay.setAlignment(QtCore.Qt.AlignHCenter |
                                                  QtCore.Qt.AlignVCenter)


if __name__ == '__main__':  # house keeping stuff, and sets style sheet and window title
    app = QApplication(sys.argv)
    window = veryCoolMan()
    # window.setStyleSheet(open("veryStylish.qss", "r").read())
    window.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window.setWindowTitle('Putt Tracker')
    window.show()
    sys.exit(app.exec())
