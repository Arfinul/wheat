import time
import cv2
import numpy as np

from util import otsu_threshold


def segment_image4(img_file, dlog=0):
    t0 = time.time()
    org = cv2.imread(img_file, cv2.IMREAD_COLOR)
    h, w = org.shape[:2]

    img = org.copy()
    print "Reading ", time.time() - t0
    t0 = time.time()

    # removing noise by using Non-local Means Denoising algorithm
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    img = cv2.resize(img, (1000, 500))
    # cv2.imshow('cleaned', img)
    print "noise removing ", time.time() - t0
    # cv2.waitKey(1)

    t0 = time.time()

    # Taking the red component out of RBG image as it is less effected by shadow of grain or impurity
    gray = np.array([[pixel[2] for pixel in row] for row in img])
    gray = cv2.resize(gray, (1000, 500))
    # cv2.imshow('gray', gray)
    # cv2.waitKey(1)

    # calculating threshold value by using otsu thresholding
    T = otsu_threshold(gray=gray)
    print "threshold calc ", time.time() - t0
    t0 = time.time()

    # generating a threshold image
    thresh = np.array([[0 if pixel < T else 255 for pixel in row] for row in gray], dtype=np.uint8)
    # thresh.save("thresholgImage.png")
    thresh = cv2.resize(thresh, (1000, 500))
    cv2.imshow('Threshold', thresh)
    cv2.waitKey(0)
    print "Generating Threshold ", time.time() - t0