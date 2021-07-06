#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 16:25:43 2021

Implementation of a program that uses image processing to detect
how many teeth the gears have.

@author: fdlopes
"""

import numpy as np
import cv2
import imutils

# Load image
img = cv2.imread('gears/gears01.png')
# Smooths the image.
smooth = cv2.blur(img, (9, 9))
# Transforms to grayscale.
gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale image with smoothing",gray)

# Binarize the image
(t,binary) = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
cv2.imshow("Binary",binary)

# Applies the sobel filter to the binarized image.
sobelX = cv2.Sobel(binary, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(binary, cv2.CV_64F, 0, 1)
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))
sobel = cv2.bitwise_or(sobelX, sobelY)
cv2.imshow("Sobel",sobel)

# Find contours in the image with sobel filter.
cnts = cv2.findContours(sobel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Get countours array
cnts = imutils.grab_contours(cnts)

# Loop over the contours
for c in cnts:
    # Compute the center of the contour
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    # Take the radius of the circle that covers the image.
    _ ,radius = cv2.minEnclosingCircle(c)
    # Paint the center of the piece white using the radius of the circle that
    # covers the piece.
    # The -20 is the number of pixels you must shrink the circle to be able to
    # capture the separate teeth of the piece.
    cv2.circle(img, (cX, cY), (int(radius)-20), (255, 255, 255), -1)
    # Show the image
    cv2.imshow("Image with center removed",img)

# Smooths the image.
smooth = cv2.blur(img, (9, 9))
# Transforms to grayscale.
gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale image with smoothing and white center",gray)

# Binarize the image
(t,binInverse) = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)

# Find all contours in the image
contours, _ = cv2.findContours(binInverse, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Displays the number of contours found, referring to the number of teeth found.
print("Number of teeth: "+str(len(contours)))
cv2.putText(binInverse,"Number of teeth: "+str(len(contours)),(0,binInverse.shape[0]),cv2.FONT_HERSHEY_DUPLEX,1,255)
cv2.imshow("Binary Inverse. Final result",binInverse)
cv2.waitKey(0)
cv2.destroyAllWindows()
