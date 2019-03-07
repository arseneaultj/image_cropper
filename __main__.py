#!/usr/bin/env python

'''
Simple "Square Detector" program.
Loads several images sequentially and tries to find squares in each image.
'''

# Python 2/3 compatibility
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2 as cv


def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img):
    # img = cv.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv.split(img):
        for thrs in xrange(0, 255, 26):
            # if thrs == 0:
            #     bin = cv.Canny(gray, 0, 50, apertureSize=5)
            #     # bin = cv.dilate(bin, None)
            # else:
            _retval, bin = cv.threshold(gray, thrs, 255, cv.THRESH_BINARY)
            contours, _hierarchy = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv.arcLength(cnt, True)
                cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv.contourArea(cnt) > 1000 and cv.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares

def find_outer_bounds(square):
    maxX =0
    maxY  = 0
    minX = 9999
    minY = 99999
    for edge in range (len(square)):
        if square[edge][0] > maxX:
            maxX = square[edge][0]
        if square[edge][0] < minX:
            minX = square[edge][0]
        if square[edge][1] > maxY:
            maxY = square[edge][1]
        if square[edge][1] < minY:
            minY = square[edge][1]
    return [maxX, maxY, minX, minY]

if __name__ == '__main__':
    from glob import glob
    # for fn in glob('../data/pic*.png'):
    for fn in glob('C:/Users/joel/PycharmProjects/image_cropper/image_cropper/testImg/*.png'):
        img = cv.imread(fn)
        # img2 = img
        squares = find_squares(img)
        
        for square in  squares:
            edges = find_outer_bounds(square)
            img2 = img[edges[3]:edges[1], edges[2]:edges[0]]
            # cv.imshow('squares',img[edges[0]:edges[2], edges[1]:[edges[3]]])
            cv.imshow('squares', img2)
            ch = cv.waitKey()
            if ch == 27:
                break
    cv.destroyAllWindows()

