import time
import cv2
import numpy as np

from Area import areaThreshold_by_havg
from _8connected import get_8connected_v2
from util import otsu_threshold, cal_segment_area

rm_detail = open('log.txt', 'a')


def segment_image4(img_file, dlog=0):
    t0 = time.time()
    org = cv2.imread(img_file, cv2.IMREAD_COLOR)
    h, w = org.shape[:2]

    img = org.copy()
    print "Reading ", time.time() - t0
    t0 = time.time()

    # removing noise by using Non-local Means Denoising algorithm
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    cv2.imwrite("1_noise_removed_image.png", img)
    # img = cv2.resize(img, (1000, 500))
    # cv2.imshow('cleaned', img)
    print "noise removing ", time.time() - t0
    # cv2.waitKey(1)

    t0 = time.time()
    # Taking the red component out of RBG image as it is less effected by shadow of grain or impurity
    gray = np.array([[pixel[2] for pixel in row] for row in img])
    cv2.imwrite("2_shadow_removed_image.png", gray)
    # gray = cv2.resize(gray, (1000, 500))
    # cv2.imshow('gray', gray)
    # cv2.waitKey(1)

    # calculating threshold value by using otsu thresholding
    T = otsu_threshold(gray=gray)
    print "threshold calc ", time.time() - t0
    t0 = time.time()

    # generating a threshold image
    thresh = np.array([[0 if pixel < T else 255 for pixel in row] for row in gray], dtype=np.uint8)
    # thresh.save("thresholgImage.png")
    # thresh = cv2.resize(thresh, (1000, 500))
    # cv2.imshow('Threshold', thresh)
    # cv2.waitKey(0)
    cv2.imwrite('3_Threshold_image.png', thresh)
    print "Generating Threshold ", time.time() - t0

    ########################## 1st level of segmentation ########################################
    print "\n Level 1 segmentation"

    # generating a mask using 8-connected component method on threshold image
    mask = get_8connected_v2(thresh, mcount=5)
    # display_mask("Initial mask",mask)
    # print "Mask Generation ",time.time()-t0
    t0 = time.time()

    # Calcutaing the grain segment using mask image
    s = cal_segment_area(mask)
    # print "Calculating segment ends",time.time()-t0
    t0 = time.time()

    # cv2.waitKey()
    # removing the backgraound of grain
    # timg = np.array([[[0,0,0] if mask[i,j] == 0 else org[i,j] for j in range(w)] for i in range(h)], dtype=np.uint8)

    # removing very small particals (smaller the 2^3 the average size)
    low_Tarea, up_Tarea = areaThreshold_by_havg(s, 3)
    slist = list(s)

    s1count = total = 0
    total += len(slist)
    for i in slist:
        area = (s[i][0] - s[i][1]) * (s[i][2] - s[i][3])
        if area < low_Tarea:  # or area > up_Tarea:
            rm = s.pop(i)
            s1count += 1
            # if dlog == 1: rm_detail.write(str(rm)+'\n')
            # cv2.imwrite('/media/zero/41FF48D81730BD9B/kisannetwork/removed/'+img_file.split('/')[-1].split(['.'])[0]+'_l1_'+str(s1count), get_img_value_inRange(org, mask, i, s[i]))
    print "\n\t%d Number of segment rejected out of %d in L1 segmentation" % (s1count, total)
    if dlog == 1:
        rm_detail.write("\n\t%d Number of segment rejected out of %d in L1 segmentation\n" % (s1count, total))
    # print " Level 1 segmentation Finished",time.time()-t0
    t0 = time.time()
    ####################### 1st level of segmentation Finished ##################################
