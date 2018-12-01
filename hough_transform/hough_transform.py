import os
import math
import numpy as np
import cv2

INPUT_DIR = 'input'
OUTPUT_DIR = 'output'
# IMG_NAME = 'lena.png'
IMG_NAME = 'capitol.png'
# IMG_NAME = 'street_signs.png'
# UPPER_THRESHOLD = 124
UPPER_THRESHOLD = 200
# UPPER_THRESHOLD = 150
JPEG_IMG_QUALITY = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
input_img_path = os.path.join(INPUT_DIR, IMG_NAME)
output_dir_path = os.path.join(OUTPUT_DIR, IMG_NAME)


def calculate_hough_accumulator(img, diagonal_length):
    acc = np.zeros((diagonal_length, 180), dtype=int)

    for y_index, y in enumerate(img):
        for x_index, x in enumerate(y):
            if x == 255:
                for theta in range(180):
                    rho = x_index * math.cos(theta) + y_index * math.sin(theta)
                    rho = int(rho)
                    acc[rho][theta] = acc[rho][theta] + 1

    return acc


def get_hough_params_of_lines(hough_accumulator, upper_threshold):
    lines_params = []
    for y_index, y in enumerate(hough_accumulator):
        for x_index, x in enumerate(y):
            if hough_accumulator[y_index][x_index] > upper_threshold:
                lines_params.append((y_index, x_index))

    return lines_params


def draw_lines(img, lines_params, line_length):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    for rho, theta in lines_params:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + line_length * (-b))
        y1 = int(y0 + line_length * (a))
        x2 = int(x0 - line_length * (-b))
        y2 = int(y0 - line_length * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return img


def save_results(img_with_lines, filename):
    output_img_path = os.path.join(output_dir_path, filename)

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    cv2.imwrite(output_img_path, img_with_lines, JPEG_IMG_QUALITY)


# main program starts here
img = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)
print('img.shape = {}'.format(img.shape))

diagonal_length = int(math.sqrt(img.shape[0] * img.shape[0] + img.shape[1] * img.shape[1]))

accumulator = calculate_hough_accumulator(img, diagonal_length)
print('Max vote for a line = {}'.format(np.amax(accumulator)))

lines_params = get_hough_params_of_lines(accumulator, UPPER_THRESHOLD)
print('Number of lines = {}'.format(len(lines_params)))

img_with_lines = draw_lines(img, lines_params, diagonal_length)

save_results(img_with_lines, 'hough_lines.jpg')

print('The end!')
