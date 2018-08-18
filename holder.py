# holder:


def track_ball_m1(self):  # this function uses findContours to track ball

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
    # vs = self.capture

    while True:

        frame = self.capture.read()
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


if self.chkTrackBall.isChecked():  # play video with ball tracking enabled

    pts = deque(maxlen=25)
    greenLower = (0, 0, 39)
    greenUpper = (359, 65, 254)

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

    while True:

        frame = self.capture.read()
        frame = frame[1]

        intCurrentFrame = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
        if intCurrentFrame >= self.endFrameNum:
            self.stop_video()

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

        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs.
        keypoints = detector.detect(mask)

        x = keypoints[0].pt[0]
        y = keypoints[0].pt[1]
        # s = keypoints[0].size  # diameter

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

        cv2.putText(frame, 'Ball 1', (int(x)+10, int(y)-10),
                    cv2.FONT_HERSHEY_SIMPLEX,  .5, (50, 255, 50), 2)

        # show the frame to our screen
        self.display_image(frame, 1)

        key = cv2.waitKey(34) & 0xFF

        if key == ord("q"):
            break
