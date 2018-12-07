import os
import cv2
import numpy as np

INPUT_DIR = 'input'
OUTPUT_DIR = 'output'
# IMG_NAME = 'lena.png'
# IMG_NAME = 'capitol.png'
# IMG_NAME = 'street_sign.jpg'
# IMG_NAME = 'opencv.jpg'
IMG_NAME = 'kitchen.jpg'
UPPER_THRESHOLD = 65
LOWER_THRESHOLD = 25
JPEG_IMG_QUALITY = [int(cv2.IMWRITE_JPEG_QUALITY), 100]


def check_hysterysis_conditions(img, current_grad_direction, grad_direction, grad_magnitude, first_neighb_y, first_neighb_x):
    if is_same_grad_directions(current_grad_direction, grad_direction[first_neighb_y][first_neighb_x]) \
    and grad_magnitude[first_neighb_y][first_neighb_x] > LOWER_THRESHOLD:

        first_neighb_magnitude = grad_magnitude[first_neighb_y][first_neighb_x]
        first_neighb_neighb_y, first_neighb_neighb_x, second_neighb_neighb_y, second_neighb_neighb_x = get_neighbours_pixel_coord_edge_opposite(
            grad_direction[first_neighb_y][first_neighb_x], first_neighb_x, first_neighb_y)

        if first_neighb_magnitude > grad_magnitude[first_neighb_neighb_y][first_neighb_neighb_x] \
        and first_neighb_magnitude > grad_magnitude[second_neighb_neighb_y][second_neighb_neighb_x]:
            if img[first_neighb_y][first_neighb_x] != 255:
                return True

    return False


def threshold_with_hysterysis(img, grad_direction, grad_magnitude):
    is_changed = False

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y][x] == 255:
                current_grad_direction = grad_direction[y][x]
                first_neighb_y, first_neighb_x, second_neighb_y, second_neighb_x = get_neighbours_pixel_coord_on_edge(
                    grad_direction[y][x], x, y)

                is_first_neighb_exists = first_neighb_y < grad_direction.shape[0] and first_neighb_x < grad_direction.shape[1]
                is_second_neighb_exists = second_neighb_y < grad_direction.shape[0] and second_neighb_x < grad_direction.shape[1]

                if not is_first_neighb_exists and not is_second_neighb_exists:
                    continue

                if is_first_neighb_exists:
                    if check_hysterysis_conditions(img, current_grad_direction, grad_direction, grad_magnitude, first_neighb_y, first_neighb_x):
                        img[first_neighb_y][first_neighb_x] = 255
                        is_changed = True

                if is_second_neighb_exists:
                    if check_hysterysis_conditions(img, current_grad_direction, grad_direction, grad_magnitude, second_neighb_y, second_neighb_x):
                        img[second_neighb_y][second_neighb_x] = 255
                        is_changed = True

    return img, is_changed


def get_neighbours_pixel_coord_on_edge(current_grad_direction, x, y):
    current_grad_direction = abs(current_grad_direction)

    if 22.5 <= current_grad_direction < 67.5:
        first_neighb_x = x + 1
        first_neighb_y = y - 1
        second_neighb_x = x - 1
        second_neighb_y = y + 1
    elif 67.5 <= current_grad_direction < 112.5:
        first_neighb_x = x + 1
        first_neighb_y = y
        second_neighb_x = x - 1
        second_neighb_y = y
    elif 112.5 <= current_grad_direction < 157.5:
        first_neighb_x = x - 1
        first_neighb_y = y - 1
        second_neighb_x = x + 1
        second_neighb_y = y + 1
    elif 157.5 <= current_grad_direction <= 180 or 0 <= current_grad_direction < 22.5:
        first_neighb_x = x
        first_neighb_y = y - 1
        second_neighb_x = x
        second_neighb_y = y + 1
    else:
        raise Exception('[ERROR] get_neighbours_pixel_coord_on_edge current_grad_direction = {}'.format(current_grad_direction))

    return first_neighb_y, first_neighb_x, second_neighb_y, second_neighb_x


def get_neighbours_pixel_coord_edge_opposite(current_grad_direction, x, y):
    current_grad_direction = abs(current_grad_direction)

    if 22.5 <= current_grad_direction < 67.5:
        first_neighb_x = x - 1
        first_neighb_y = y - 1
        second_neighb_x = x + 1
        second_neighb_y = y + 1
    elif 67.5 <= current_grad_direction < 112.5:
        first_neighb_x = x
        first_neighb_y = y - 1
        second_neighb_x = x
        second_neighb_y = y + 1
    elif 112.5 <= current_grad_direction < 157.5:
        first_neighb_x = x + 1
        first_neighb_y = y - 1
        second_neighb_x = x - 1
        second_neighb_y = y + 1
    elif 157.5 <= current_grad_direction <= 180 or 0 <= current_grad_direction < 22.5:
        first_neighb_x = x - 1
        first_neighb_y = y
        second_neighb_x = x + 1
        second_neighb_y = y
    else:
        raise Exception('[ERROR] get_neighbours_pixel_coord_edge_opposite current_grad_direction = {}'.format(current_grad_direction))

    return first_neighb_y, first_neighb_x, second_neighb_y, second_neighb_x


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


def non_maximum_suppression(img, grad_magnitude, grad_direction):
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            current_grad_magnitude = grad_magnitude[y][x]
            current_grad_direction = grad_direction[y][x]

            first_neighb_y, first_neighb_x, second_neighb_y, second_neighb_x = get_neighbours_pixel_coord_edge_opposite(current_grad_direction, x, y)

            if first_neighb_x >= img.shape[1] or first_neighb_y >= img.shape[0]:
                first_neighb_magnitude = 0
            else:
                first_neighb_magnitude = grad_magnitude[first_neighb_y][first_neighb_x]

            if second_neighb_x >= img.shape[1] or second_neighb_y >= img.shape[0]:
                second_neighb_magnitude = 0
            else:
                second_neighb_magnitude = grad_magnitude[second_neighb_y][second_neighb_x]

            if current_grad_magnitude > first_neighb_magnitude\
            and current_grad_magnitude > second_neighb_magnitude\
            and current_grad_magnitude > UPPER_THRESHOLD:
                    img[y][x] = 255
            else:
                img[y][x] = 0

    return img


input_img_path = os.path.join(INPUT_DIR, IMG_NAME)
output_dir_path = os.path.join(OUTPUT_DIR, IMG_NAME)

if not os.path.exists(output_dir_path):
    os.makedirs(output_dir_path)

img = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)
print('img.shape = {}'.format(img.shape))

blurred_img = cv2.GaussianBlur(img, (5, 5), 1.4)
cv2.imwrite(os.path.join(output_dir_path, 'gaussian_blur.jpg'), blurred_img, JPEG_IMG_QUALITY)

opencv_canny = cv2.Canny(blurred_img, LOWER_THRESHOLD, UPPER_THRESHOLD)
cv2.imwrite(os.path.join(output_dir_path, 'opencv_canny.jpg'), opencv_canny, JPEG_IMG_QUALITY)

sobel_x = cv2.Sobel(blurred_img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred_img, cv2.CV_64F, 0, 1, ksize=3)

cv2.imwrite(os.path.join(output_dir_path, 'sobel_x.jpg'), sobel_x, JPEG_IMG_QUALITY)
cv2.imwrite(os.path.join(output_dir_path, 'sobel_y.jpg'), sobel_y, JPEG_IMG_QUALITY)

grad_magnitude = np.sqrt(np.add(np.power(sobel_x, 2), np.power(sobel_y, 2)))
cv2.imwrite(os.path.join(output_dir_path, 'grad_magnitude.jpg'), grad_magnitude, JPEG_IMG_QUALITY)
grad_magnitude_img = cv2.imread(os.path.join(output_dir_path, 'grad_magnitude.jpg'), cv2.IMREAD_GRAYSCALE)
print('grad_magnitude_img max = {}'.format(np.amax(grad_magnitude_img)))

grad_direction = np.arctan2(sobel_y, sobel_x) * 180 / np.pi

print('grad_magnitude.shape = {}'.format(grad_magnitude.shape))
print('sobel_x max = {}'.format(np.amax(sobel_x)))
print('grad_magnitude max = {}'.format(np.amax(grad_magnitude)))
print('grad_magnitude min = {}'.format(np.amin(grad_magnitude)))

non_maximum_suppressed_img = non_maximum_suppression(grad_magnitude_img, grad_magnitude, grad_direction)

cv2.imwrite(os.path.join(output_dir_path, 'non_maximum_suppressed_img.jpg'), non_maximum_suppressed_img, JPEG_IMG_QUALITY)

thresholded_with_hysterysis_img, is_changed = threshold_with_hysterysis(non_maximum_suppressed_img, grad_direction, grad_magnitude)

i = 0
while is_changed:
    thresholded_with_hysterysis_img, is_changed = threshold_with_hysterysis(thresholded_with_hysterysis_img,
                                                                            grad_direction,
                                                                            grad_magnitude)

    if i % 10 == 0:
        cv2.imwrite(os.path.join(output_dir_path, 'thresholded_with_hysterysis_img_{}.jpg'.format(i)),
                    thresholded_with_hysterysis_img, JPEG_IMG_QUALITY)

    i = i + 1
    print('Thresholded with Hysterysis: {}'.format(i))

cv2.imwrite(os.path.join(output_dir_path,
            'thresholded_with_hysterysis_img_{}.jpg'.format(i)),
            thresholded_with_hysterysis_img,
            JPEG_IMG_QUALITY)
