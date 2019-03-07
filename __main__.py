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
    gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img = cv.GaussianBlur(gray_img, (5, 5), 0)
    # _, threah_img = cv.threshold(img,127,255,)
    # img = cv.medianBlur(gray_img,5)
    threah_img = cv.adaptiveThreshold(img, 100, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 27, 9)
    cv.imshow("thres", threah_img)
    squares = []
    canny = cv.Canny(threah_img, 127, 255)
    cv.imshow("canny", canny)
    contours, _hierarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cnt_len = cv.arcLength(cnt, True)
        cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True)
        if len(cnt) == 4 and cv.contourArea(cnt) > 100 and cv.isContourConvex(cnt):
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


def convert_to_png():

    from glob import glob
    jpgs = glob('../**/*.jpg', recursive=True)
    for j in jpgs:
        img = cv.imread(j)
        cv.imwrite(j[:-3] + 'png', img)

def rotate_image(imgpath):
    img = cv.imread(imgpath, 0)
    rows, cols = img.shape

    M = cv.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    dst = cv.warpAffine(img, M, (cols, rows))

if __name__ == '__main__':

    from glob import glob

    for fn in glob('./images/**/*.png', recursive=True):
        # i = 90
        # while(i<360):
        #     img = cv.imread(fn,0)
        #     rows, cols = img.shape
        #
        #     M = cv.getRotationMatrix2D((cols / 2, rows / 2), i, 1)
        #     dst = cv.warpAffine(img, M, (cols, rows))
        #     cv.imwrite(fn[:-4] + '-' + str(i) + '.png', dst)
        #     i += 90

        img = cv.imread(fn)
        # img2 = img
        squares = find_squares(img)
        cv.drawContours(img,squares, -1,(0,255,0),3)
        cv.imshow('squares', img)
        ch = cv.waitKey()
        if ch == 27:
            break

        for square in  squares:
            edges = find_outer_bounds(square)
            img2 = img[edges[3]:edges[1], edges[2]:edges[0]]
            # cv.imshow('squares',img[edges[0]:edges[2], edges[1]:[edges[3]]])
            cv.imshow('squares', img2)
            ch = cv.waitKey()
            if ch == 27:
                break
    cv.destroyAllWindows()
