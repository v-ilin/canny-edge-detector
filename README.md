# The Canny Edge Detector
This is implementation of edge detection on images by algorithm developed by John F. Canny.

Requirements:
1. Python (2.7 or 3)
2. OpenCV

## Algorithm

A full explanation of algorithm and implementation on C++ can be found [here](http://aishack.in/tutorials/canny-edge-detector/)

The process of edge detection consists of following steps:
1. Non-maximum suppression
2. Thresholding with Hysterysis

### Non-maximum suppression

1. Calculating first derivatives by x and y
2. Calucating gradiend magnitude and direction
3. Filtering pixels by upper threshold of gradient magnitude
4. Suppression pixels which are not maximum in a local area (comparison with it's neighbours)

### Thresholding with Hysterysis

1. Filtering pixels by lower threshold of gradient magnitude
2. Completing the edges
