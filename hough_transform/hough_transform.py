import os
import math
import numpy as np
import cv2

INPUT_DIR = 'input'
IMG_NAME = 'lena.png'
JPEG_IMG_QUALITY = [int(cv2.IMWRITE_JPEG_QUALITY), 100]


input_img_path = os.path.join(INPUT_DIR, IMG_NAME)
img = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)
print('img.shape = {}'.format(img.shape))

diagonal_length = len(img.diagonal())

acc = np.zeros((diagonal_length, 180), dtype=int)

for y_index, y in enumerate(img):
    for x_index, x in enumerate(y):
        if x == 255:
            for angle in range(180):
                r = x_index * math.cos(angle)+ y_index * math.sin(angle)
                r = int(r)
                acc[r][angle] = acc[r][angle] + 1

print('The end!')
