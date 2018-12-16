import os
import math
import numpy as np
import cv2

from common import utils

INPUT_DIR = 'hough_transform' + os.sep + 'input'
OUTPUT_DIR = 'hough_transform' + os.sep + 'output'
# IMG_NAME = 'lena.jpg'
# IMG_NAME = 'capitol.jpg'
# IMG_NAME = 'street_signs.jpg'
IMG_NAME = 'circles.jpg'
UPPER_THRESHOLD_LINES = 80
# UPPER_THRESHOLD_LINES = 200
# UPPER_THRESHOLD_LINES = 150
UPPER_THRESHOLD_CIRCLES = 12
JPEG_IMG_QUALITY = [int(cv2.IMWRITE_JPEG_QUALITY), 100]


def calculate_hough_lines_accumulator(img, diagonal_length):
    acc = np.zeros((diagonal_length, 180), dtype=int)

    for y_index, y in enumerate(img):
        for x_index, x in enumerate(y):
            if x == 255:
                for theta in range(180):
                    rho = x_index * math.cos(theta) + y_index * math.sin(theta)
                    rho = int(rho)
                    acc[rho][theta] = acc[rho][theta] + 1

    return acc


def calculate_circle_radius(x, y, x_center, y_center):
    return int(math.sqrt(pow(x - x_center, 2) + pow(y - y_center, 2)))


def calculate_hough_circles_accumulator(img):
    R_max = 50

    acc = np.zeros((500, 500, R_max + 1), dtype=int)

    for y_index, y in enumerate(img):
        for x_index, x in enumerate(y):
            if x == 255:
                for r in range(R_max):
                    for theta in range(0, 361, 30):
                        a = int(x_index - r * math.cos(theta * math.pi / 180.0))
                        b = int(y_index - r * math.sin(theta * math.pi / 180.0))
                        acc[b][a][r] = acc[b][a][r] + 1

        print('y = {}/{}'.format(y_index, img.shape[0]))

    return acc


def filter_lines(hough_accumulator, upper_threshold):
    lines_params = []
    for y_index, y in enumerate(hough_accumulator):
        for x_index, x in enumerate(y):
            if hough_accumulator[y_index][x_index] > upper_threshold:
                lines_params.append((y_index, x_index))

    return lines_params


def filter_circles(hough_accumulator, upper_threshold):
    circles_params = []
    for y_index, y in enumerate(hough_accumulator):
        for x_index, x in enumerate(y):
            for R_index, R in enumerate(x):
                if hough_accumulator[y_index][x_index][R_index] > upper_threshold:
                    circles_params.append((y_index, x_index, R_index))

    return circles_params


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


def draw_circles(img, circles_params):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    for y, x, R in circles_params:
        cv2.circle(img, (x, y), R, (0, 0, 255), 2)

    return img


def hough_lines_transform(img):
    diagonal_length = int(math.sqrt(img.shape[0] * img.shape[0] + img.shape[1] * img.shape[1]))

    accumulator = calculate_hough_lines_accumulator(img, diagonal_length)
    print('Max vote for a line = {}'.format(np.amax(accumulator)))

    lines_params = filter_lines(accumulator, UPPER_THRESHOLD_LINES)
    print('Number of lines = {}'.format(len(lines_params)))

    img_with_lines = draw_lines(img, lines_params, diagonal_length)

    output_dir = os.path.join(OUTPUT_DIR, IMG_NAME)
    saved_image_path = utils.save_image(output_dir, img_with_lines, 'hough_lines.jpg')

    return saved_image_path


def hough_circles_transform(img):
    accumulator = calculate_hough_circles_accumulator(img)
    print('Max vote for a circle = {}'.format(np.amax(accumulator)))

    circles_params = filter_circles(accumulator, UPPER_THRESHOLD_CIRCLES)
    print('Number of circles = {}'.format(len(circles_params)))

    img_with_circles = draw_circles(img, circles_params)

    output_dir = os.path.join(OUTPUT_DIR, IMG_NAME)
    saved_image_path = utils.save_image(output_dir, img_with_circles, 'hough_circles.jpg')

    return saved_image_path


def main():
    input_img_path = os.path.join(INPUT_DIR, IMG_NAME)
    img = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)
    print('img.shape = {}'.format(img.shape))

    saved_image_with_lines = hough_lines_transform(img)
    saved_image_with_circles = hough_circles_transform(img)

    print('A result saved to: {}'.format(saved_image_with_lines))
    print('A result saved to: {}'.format(saved_image_with_circles))


if __name__ == '__main__':
    main()