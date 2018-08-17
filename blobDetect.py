# Standard imports
import cv2
import numpy as np

# Read image
im = cv2.imread("testBRO.jpeg", cv2.IMREAD_GRAYSCALE)


im = cv2.bitwise_not(im)
# im = cv2.GaussianBlur(im, (23, 23), 0)

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10
params.maxThreshold = 150


# Filter by Area.
params.filterByArea = True
params.minArea = 15

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3:
    detector = cv2.SimpleBlobDetector(params)
else:
    detector = cv2.SimpleBlobDetector_create(params)


# Detect blobs.
keypoints = detector.detect(im)
print(keypoints)

x = keypoints[0].pt[0]
y = keypoints[0].pt[1]
s = keypoints[0].size  # diameter
print(x)
print(y)
print(s)
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array(
    []), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show blobs
cv2.circle(im_with_keypoints, (int(x), int(y)), 2, (0, 0, 255), -1)
cv2.putText(im_with_keypoints, 'Ball 1', (int(x)+10, int(y)-10),
            cv2.FONT_HERSHEY_SIMPLEX,  .5, (50, 255, 50), 2)
cv2.imshow("Keypoints", im_with_keypoints)

cv2.waitKey(0)
