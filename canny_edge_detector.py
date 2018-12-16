import os
import cv2
import numpy as np

from common import utils

INPUT_DIR = 'canny_edge_detector' + os.sep + 'input'
OUTPUT_DIR = 'canny_edge_detector' + os.sep + 'output'
IMG_NAME = 'lena.jpg'
# IMG_NAME = 'capitol.png'
# IMG_NAME = 'street_sign.jpg'
# IMG_NAME = 'opencv.jpg'
# IMG_NAME = 'kitchen.jpg'
UPPER_THRESHOLD = 55
LOWER_THRESHOLD = 25


def check_hysterysis_conditions(img, current_grad_direction, grad_direction, grad_magnitude, y_neighb, x_neighb):
    grad_direction_neighb = grad_direction[y_neighb][x_neighb]

    if is_same_grad_directions(current_grad_direction, grad_direction_neighb) \
    and grad_magnitude[y_neighb][x_neighb] > LOWER_THRESHOLD:

        magnitude_1 = grad_magnitude[y_neighb][x_neighb]
        y1, x1, y2, x2 = get_neighbours_coord_across_edge(
            grad_direction[y_neighb][x_neighb], x_neighb, y_neighb)

        if magnitude_1 > grad_magnitude[y1][x1] \
        and magnitude_1 > grad_magnitude[y2][x2]:
            if img[y_neighb][x_neighb] != 255:
                return True

    return False


def threshold_with_hysterysis(img, grad_direction, grad_magnitude):
    if grad_magnitude.shape != grad_direction.shape:
        raise ValueError('Gradient magnitudes and directions must have one shape!')

    is_changed = False

    x_max = grad_magnitude.shape[1] - 1
    y_max = grad_magnitude.shape[0] - 1

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y][x] == 255:
                current_grad_direction = grad_direction[y][x]
                y1, x1, y2, x2 = get_neighbours_coord_along_edge(grad_direction[y][x], x, y)

                is_first_neighb_exists = y1 <= y_max and x1 <= x_max \
                                         and y1 >= 0 and x1 >= 0

                is_second_neighb_exists = y2 <= y_max and x2 <= x_max \
                                          and y2 >= 0 and x2 >= 0

                if not is_first_neighb_exists and not is_second_neighb_exists:
                    continue

                if is_first_neighb_exists:
                    if check_hysterysis_conditions(img, current_grad_direction, grad_direction, grad_magnitude, y1, x1):
                        img[y1][x1] = 255
                        is_changed = True

                if is_second_neighb_exists:
                    if check_hysterysis_conditions(img, current_grad_direction, grad_direction, grad_magnitude, y2, x2):
                        img[y2][x2] = 255
                        is_changed = True

    return img, is_changed


def get_neighbours_coord_along_edge(current_grad_direction, x, y):
    current_grad_direction = abs(current_grad_direction)

    if 22.5 <= current_grad_direction < 67.5:
        x1 = x + 1
        y1 = y - 1
        x2 = x - 1
        y2 = y + 1
    elif 67.5 <= current_grad_direction < 112.5:
        x1 = x + 1
        y1 = y
        x2 = x - 1
        y2 = y
    elif 112.5 <= current_grad_direction < 157.5:
        x1 = x - 1
        y1 = y - 1
        x2 = x + 1
        y2 = y + 1
    elif 157.5 <= current_grad_direction <= 180 or 0 <= current_grad_direction < 22.5:
        x1 = x
        y1 = y - 1
        x2 = x
        y2 = y + 1
    else:
        raise Exception('[ERROR] get_neighbours_pixel_coord_on_edge current_grad_direction = {}'.format(current_grad_direction))

    return y1, x1, y2, x2


def get_neighbours_coord_across_edge(current_grad_direction, x, y):
    current_grad_direction = abs(current_grad_direction)

    if 22.5 <= current_grad_direction < 67.5:
        x1 = x - 1
        y1 = y - 1
        x2 = x + 1
        y2 = y + 1
    elif 67.5 <= current_grad_direction < 112.5:
        x1 = x
        y1 = y - 1
        x2 = x
        y2 = y + 1
    elif 112.5 <= current_grad_direction < 157.5:
        x1 = x + 1
        y1 = y - 1
        x2 = x - 1
        y2 = y + 1
    elif 157.5 <= current_grad_direction <= 180 or 0 <= current_grad_direction < 22.5:
        x1 = x - 1
        y1 = y
        x2 = x + 1
        y2 = y
    else:
        raise Exception('[ERROR] get_neighbours_pixel_coord_edge_opposite current_grad_direction = {}'.format(current_grad_direction))

    return y1, x1, y2, x2


def is_same_grad_directions(direction1, direction2):
    direction1 = abs(direction1)
    direction2 = abs(direction2)

    if 0 <= direction1 < 22.5 and 0 <= direction2 < 22.5:
        return True
    elif 22.5 <= direction1 < 67.5 and 22.5 <= direction2 < 67.5:
        return True
    elif 67.5 <= direction1 < 112.5 and 67.5 <= direction2 < 112.5:
        return True
    elif 112.5 <= direction1 < 157.5 and 112.5 <= direction2 < 157.5:
        return True
    elif 157.5 <= direction1 <= 180 and 157.5 <= direction2 <= 180:
        return True
    else:
        return False


def suppress_not_max_pixels(grad_magnitude, grad_direction):
    if grad_magnitude.shape != grad_direction.shape:
        raise ValueError('Gradient magnitudes and directions must have one shape!')

    result = np.zeros(grad_magnitude.shape, dtype=np.uint8)

    x_max = grad_magnitude.shape[1] - 1
    y_max = grad_magnitude.shape[0] - 1

    for y in range(grad_magnitude.shape[0]):
        for x in range(grad_magnitude.shape[1]):
            current_grad_magnitude = grad_magnitude[y][x]

            if current_grad_magnitude < UPPER_THRESHOLD:
                continue

            current_grad_direction = grad_direction[y][x]

            y1, x1, y2, x2 = get_neighbours_coord_across_edge(current_grad_direction, x, y)

            if x1 > x_max or y1 > y_max or x1 < 0 or y1 < 0:
                magnitude_1 = 0
            else:
                magnitude_1 = grad_magnitude[y1][x1]

            if x2 > x_max or y2 > y_max or x2 < 0 or y2 < 0:
                magnitude_2 = 0
            else:
                magnitude_2 = grad_magnitude[y2][x2]

            if current_grad_magnitude > max(magnitude_1, magnitude_2):
                result[y][x] = 255

    return result


def main():
    input_img_path = os.path.join(INPUT_DIR, IMG_NAME)
    output_dir_path = os.path.join(OUTPUT_DIR, IMG_NAME)

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    img = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)
    blurred_img = cv2.GaussianBlur(img, (5, 5), 1.4)

    opencv_canny = cv2.Canny(blurred_img, LOWER_THRESHOLD, UPPER_THRESHOLD)

    sobel_x = cv2.Sobel(blurred_img, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(blurred_img, cv2.CV_64F, 0, 1)

    grad_magnitude = np.sqrt(np.add(np.power(sobel_x, 2), np.power(sobel_y, 2)))
    grad_direction = np.arctan2(sobel_y, sobel_x) * 180.0 / np.pi

    img_max_suppressed = suppress_not_max_pixels(grad_magnitude, grad_direction)

    utils.save_image(output_dir_path, blurred_img, 'gaussian_blur.jpg')
    utils.save_image(output_dir_path, grad_magnitude, 'grad_magnitude.jpg')
    utils.save_image(output_dir_path, sobel_x, 'sobel_x.jpg')
    utils.save_image(output_dir_path, sobel_y, 'sobel_y.jpg')
    utils.save_image(output_dir_path, opencv_canny, 'opencv_canny.jpg')
    utils.save_image(output_dir_path, img_max_suppressed, 'non_maximum_suppressed_img.jpg')

    print('img.shape = {}'.format(img.shape))
    print('grad_magnitude_img max = {}'.format(np.amax(grad_magnitude)))
    print('grad_magnitude.shape = {}'.format(grad_magnitude.shape))
    print('sobel_x max = {}'.format(np.amax(sobel_x)))
    print('grad_magnitude max = {}'.format(np.amax(grad_magnitude)))
    print('grad_magnitude min = {}'.format(np.amin(grad_magnitude)))

    img_thresholded_with_hysterysis, is_changed = threshold_with_hysterysis(
        img_max_suppressed, grad_direction, grad_magnitude)

    i = 0
    while is_changed:
        img_thresholded_with_hysterysis, is_changed = threshold_with_hysterysis(
            img_thresholded_with_hysterysis,
            grad_direction,
            grad_magnitude)

        if i % 10 == 0:
            utils.save_image(output_dir_path, img_thresholded_with_hysterysis, 'thresholded_with_hysterysis_img_{}.jpg'.format(i))

        i = i + 1
        print('Thresholded with Hysterysis: {}'.format(i))

    utils.save_image(output_dir_path, img_thresholded_with_hysterysis, 'thresholded_with_hysterysis_img_{}.jpg'.format(i))


if __name__ == '__main__':
    main()
