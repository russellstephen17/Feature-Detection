# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 10:25:21 2020

@author: srussel1
"""

import numpy as np
import cv2
import matplotlib
import math
import time
from codecs import decode
import struct
#from matplotlib import pyplot as plt

from skimage.feature import corner_harris, peak_local_max
from scipy import ndimage

def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def anms(coords, top=400):
    l, x, y = [], 0, 0
    while x < len(coords):
        minpoint = 99999999
        xi = coords[x].pt[0]
        yi =  coords[x].pt[1]
        while y < len(coords):   
            xj, yj = coords[y].pt[0], coords[y].pt[1]
            if (xi != xj and yi != yj) and coords[x].response < 0.9 * coords[y].response:
                dist = distance(xi, yi, xj, yj)
                if dist < minpoint:
                    minpoint = dist
            y += 1
        l.append([xi, yi, minpoint])
        x += 1
        y = 0
    l.sort(key=lambda x: x[2])
    l = l[0:top]
    #print l
    return l

def int_to_bytes(n, length):  # Helper function
    """ Int/long to byte string.

        Python 3.2+ has a built-in int.to_bytes() method that could be used
        instead, but the following works in earlier versions including 2.x.
    """
    return decode('%%0%dx' % (length << 1) % n, 'hex')[-length:]

def bin_to_float(b):
    """ Convert binary string to a float. """
    bf = int_to_bytes(int(b, 2), 8)  # 8 bytes needed for IEEE 754 binary64.
    return struct.unpack('>d', bf)[0]

def get_harris_corners(im, edge_discard=20):
    """
    This function takes a b&w image and an optional amount to discard
    on the edge (default is 5 pixels), and finds all harris corners
    in the image. Harris corners near the edge are discarded and the
    coordinates of the remaining corners are returned. A 2d array (h)
    containing the h value of every pixel is also returned.

    h is the same shape as the original image, im.
    coords is 2 x n (ys, xs).
    """

    assert edge_discard >= 20

    # find harris corners
    h = corner_harris(im, method='eps', sigma=1)
    coords = peak_local_max(h, min_distance=1, indices=True)

    # discard points on edge
    edge = edge_discard  # pixels
    mask = (coords[:, 0] > edge) & \
           (coords[:, 0] < im.shape[0] - edge) & \
           (coords[:, 1] > edge) & \
           (coords[:, 1] < im.shape[1] - edge)
    coords = coords[mask].T
    return h, coords



def corners_to_keypoints(corners):
    if corners is None:
        keypoints = []
    else:
        keypoints = [cv2.KeyPoint(kp[0][0], kp[0][1], 1) for kp in corners]
            
    return keypoints

orb = cv2.ORB_create()
brisk = cv2.BRISK_create()
fast = cv2.FastFeatureDetector_create()
sift = cv2.xfeatures2d.SIFT_create()
img1 = cv2.imread('Cath1.jpg')
gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

img2 = cv2.imread('Cath2.jpg')
#img2 = ndimage.rotate(img1, 45)
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

M = max([img1.shape[0], img2.shape[0]])
N = img1.shape[1] + img2.shape[1]
img_match = np.zeros((M, N))

tic = time.perf_counter()
# Shi Tomasi
#ShiTomasiFeats1 = cv2.goodFeaturesToTrack(gray1,1000,0.05,10)
#keypoints = corners_to_keypoints(ShiTomasiFeats1)
#imgShi1 = cv2.drawKeypoints(img1, keypoints, None, color=(0,0,255))
#
#ShiTomasiFeats2 = cv2.goodFeaturesToTrack(gray2,1000,0.05,10)
#keypoints2 = corners_to_keypoints(ShiTomasiFeats2)
#imgShi2 = cv2.drawKeypoints(img2, keypoints2, None, color=(0,0,255))


# Harris Stephens
HarrisStephenFeats1 = cv2.cornerHarris(gray1,2,3,0.06)
#print(ShiTomasiFeats)
ret, HarrisStephenFeats1 = cv2.threshold(HarrisStephenFeats1,0.1*HarrisStephenFeats1.max(),255,0)
HarrisStephenFeats1 = np.uint8(HarrisStephenFeats1)
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(HarrisStephenFeats1)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray1,np.float32(centroids),(5,5),(-1,-1),criteria)
keypoints = [cv2.KeyPoint(crd[0], crd[1], 13) for crd in corners]

HarrisStephenFeats2 = cv2.cornerHarris(gray2,2,3,0.06)
#print(ShiTomasiFeats)
ret2, HarrisStephenFeats2 = cv2.threshold(HarrisStephenFeats2,0.1*HarrisStephenFeats2.max(),255,0)
HarrisStephenFeats2 = np.uint8(HarrisStephenFeats2)
ret2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(HarrisStephenFeats2)
criteria2 = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners2 = cv2.cornerSubPix(gray2,np.float32(centroids2),(5,5),(-1,-1),criteria2)
keypoints2 = [cv2.KeyPoint(crd[0], crd[1], 13) for crd in corners2]

imgHarr1 = cv2.drawKeypoints(img1, keypoints, None, color=(0,0,255))

#det1 =orb.detect(img1)
#det2 =orb.detect(img2)

# brisk
#keypoints =brisk.detect(img1)
#keypoints2 =brisk.detect(img2)
#keypoints, des1 = brisk.compute(img1, keypoints)
#imgBrisk1 = cv2.drawKeypoints(img1, keypoints, None, color=(0,0,255))
#print(keypoints[0].response)

#keypoints =fast.detect(img1)
#keypoints2 =fast.detect(img2)
#imgFast1 = cv2.drawKeypoints(img1, keypoints, None, color=(0,0,255))
#
#
#modified1 = anms(keypoints, top=1000)
#modified2 = anms(keypoints2, top=1000)
#modified11 = [cv2.KeyPoint(crd[0], crd[1], 13) for crd in modified1]
#modified21 = [cv2.KeyPoint(crd[0], crd[1], 13) for crd in modified2]
#imgBriskmod1 = cv2.drawKeypoints(img1, modified11, None, color=(0,0,255))
#
#
#numpy_horizontal = np.hstack((imgBrisk1, imgBriskmod1))
#numpy_horizontal_concat = np.concatenate((imgBrisk1, imgFast1), axis=1)
#keypointsST1, des1 = orb.compute(img1, keypointsST1)
#keypointsST2, des2 = orb.compute(img2, keypointsST2)
#corner_keypoints, des1 = orb.compute(img1, corner_keypoints)
#corner_keypoints2, des2 = orb.compute(img2, corner_keypoints2)
#keypoints = sift.detect(img1)
#keypoints2 = sift.detect(img2)
#
keypoints, des1 = sift.compute(img1, keypoints)
keypoints2, des2 = sift.compute(img2, keypoints2)
#imgSift1 = cv2.drawKeypoints(img1, keypoints, None, color=(0,0,255))
#keypoints = brisk.detect(img1)
#keypoints2 = brisk.detect(img2)
#
#keypoints, des1 = brisk.compute(img1, keypoints)
#keypoints2, des2 = brisk.compute(img2, keypoints2)
#
#imgBrisk1 = cv2.drawKeypoints(img1, keypoints, None, color=(0,0,255))
#numpy_horizontal = np.hstack((imgHarr1, imgShi1))
#des3=bin_to_float(des1)
#print(des1)
# FLANN parameters

#FLANN_INDEX_KDTREE = 0
#index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#search_params = dict(checks=50)   # or pass empty dictionary
#
#flann = cv2.FlannBasedMatcher(index_params,search_params)
##if(des1.type()!=CV_32F):
##    des1.convertTo(des1, CV_32F)
##
##if(des2.type()!=CV_32F):
##    des2.convertTo(des2, CV_32F)
#    
#matches = flann.knnMatch(des1,des2,k=2)
#
## Need to draw only good matches, so create a mask
#matchesMask = [[0,0] for i in range(len(matches))]
#
## ratio test as per Lowe's paper
#for i,(m,n) in enumerate(matches):
#    if m.distance < 0.7*n.distance:
#        matchesMask[i]=[1,0]
#
#draw_params = dict(matchColor = (0,255,0),
#                   singlePointColor = (255,0,0),
#                   matchesMask = matchesMask,
#                   flags = 0)
#
#img3 = cv2.drawMatchesKnn(img1,keypointsST1,img2,keypointsST2,matches,None,**draw_params)
#cv2.imshow('match',imgSift1)
#cv2.imshow('Numpy Horizontal', numpy_horizontal)
#if cv2.waitKey(0) & 0xff == 27:
#    cv2.destroyAllWindows()

# FLANN parameters
#FLANN_INDEX_KDTREE = 0
#index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#search_params = dict(checks=50)   # or pass empty dictionary
#
#flann = cv2.FlannBasedMatcher(index_params,search_params)
#
#matches = flann.knnMatch(des1,des2,k=2)
#good = []
#for m,n in matches:
#    if m.distance < 0.7*n.distance:
#        good.append([m])
## Need to draw only good matches, so create a mask
##matchesMask = [[0,0] for i in range(len(matches))]
##
### ratio test as per Lowe's paper
##for i,(m,n) in enumerate(matches):
##    if m.distance < 0.7*n.distance:
##        matchesMask[i]=[1,0]
##
##draw_params = dict(matchColor = (0,255,0),
##                   singlePointColor = (255,0,0),
##                   matchesMask = matchesMask,
##                   flags = 0)
#
##img3 = cv2.drawMatchesKnn(img1,keypoints,img2,keypoints2,matches,None,**draw_params)
#img3 = cv2.drawMatchesKnn(img1,keypoints,img2,keypoints2,good, img_match, flags=2)
#cv2.imshow('match',img3)
#if cv2.waitKey(0) & 0xff == 27:
#    cv2.destroyAllWindows()
toc = time.perf_counter()
print(f"Time elapsed {toc - tic:0.4f} seconds")


## BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,keypoints,img2,keypoints2,good, img_match, flags=2)
#cv2.imshow('match',img3)
#if cv2.waitKey(0) & 0xff == 27:
#    cv2.destroyAllWindows()













#imgHS1 = cv2.drawKeypoints(img1, corner_keypoints, None, color=(0,0,255))


# compute the descriptors with ORB
#corner_keypoints, desST1 = orb.compute(img1, corner_keypoints)

#output = np.hstack([imgShi1, imgShi2])
#cv2.imshow('Shi Tomasi1', imgShi1)
#cv2.imshow('Shi Tomasi2', imgShi2)
#if cv2.waitKey(0) & 0xff == ord(' '):
#    cv2.destroyAllWindows()
    
