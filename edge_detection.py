import os
import cv2
import numpy as np

input_dir = 'input'
output_dir_name = 'output'
img_name = "lena.png"
threshold = 70


def threshold_with_hysterysis(img, grad_direction, grad_magnitude):
    is_changed = False

    for y in range(0, img.shape[0], 1):
        for x in range(0, img.shape[1], 1):
            if img[y][x] == 255:
                current_grad_direction = grad_direction[y][x]
                first_neighb_y, first_neighb_x, second_neighb_y, second_neighb_x = get_neighbours_pixel_coord_on_edge(
                    grad_direction[y][x], x, y)

                is_first_neighb_exists = first_neighb_y < grad_direction.shape[0] and first_neighb_x < grad_direction.shape[1]
                is_second_neighb_exists = second_neighb_y < grad_direction.shape[0] and second_neighb_x < grad_direction.shape[1]

                if is_first_neighb_exists:
                    if is_same_grad_directions(current_grad_direction, grad_direction[first_neighb_y][first_neighb_x]):
                        if img[first_neighb_y][first_neighb_x] != 255:
                            img[first_neighb_y][first_neighb_x] = 255
                            is_changed = True

                    if grad_magnitude[first_neighb_y][first_neighb_x] > threshold:
                        if img[first_neighb_y][first_neighb_x] != 255:
                            img[first_neighb_y][first_neighb_x] = 255
                            is_changed = True

                    first_neighb_magnitude = grad_magnitude[first_neighb_y][first_neighb_x]
                    first_neighb_neighb_y, first_neighb_neighb_x, second_neighb_neighb_y, second_neighb_neighb_x = get_neighbours_pixel_coord_edge_opposite(
                        grad_direction[first_neighb_y][first_neighb_x], first_neighb_x, first_neighb_y)

                    if first_neighb_magnitude > grad_magnitude[first_neighb_neighb_y][first_neighb_neighb_x] and first_neighb_magnitude > grad_magnitude[second_neighb_neighb_y][second_neighb_neighb_x]:
                        if img[first_neighb_y][first_neighb_x] != 255:
                            img[first_neighb_y][first_neighb_x] = 255
                            is_changed = True

                if is_second_neighb_exists:
                    if is_same_grad_directions(current_grad_direction, grad_direction[second_neighb_y][second_neighb_x]):
                        if img[second_neighb_y][second_neighb_x] != 255:
                            img[second_neighb_y][second_neighb_x] = 255
                            is_changed = True

                    if grad_magnitude[second_neighb_y][second_neighb_x] > threshold:
                        if img[second_neighb_y][second_neighb_x] != 255:
                            img[second_neighb_y][second_neighb_x] = 255
                            is_changed = True

                    second_neighb_magnitude = grad_magnitude[second_neighb_y][second_neighb_x]
                    first_neighb_neighb_y, first_neighb_neighb_x, second_neighb_neighb_y, second_neighb_neighb_x = get_neighbours_pixel_coord_edge_opposite(
                        grad_direction[second_neighb_y][second_neighb_x], second_neighb_x, second_neighb_y)

                    if second_neighb_magnitude > grad_magnitude[first_neighb_neighb_y][first_neighb_neighb_x] and second_neighb_magnitude > grad_magnitude[second_neighb_neighb_y][second_neighb_neighb_x]:
                        if img[second_neighb_y][second_neighb_x] != 255:
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
        print('[ERROR] get_neighbours_pixel_coord_on_edge current_grad_direction = {}'.format(current_grad_direction))

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
        print('[ERROR] get_neighbours_pixel_coord_edge_opposite current_grad_direction = {}'.format(current_grad_direction))

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
    for y in range(0, img.shape[0], 1):
        for x in range(0, img.shape[1], 1):
            current_grad_magnitute = grad_magnitude[y][x]
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

            if current_grad_magnitute > first_neighb_magnitude and current_grad_magnitute > second_neighb_magnitude:
                # print('current_magnitute = {}'.format(current_magnitute))

                if current_grad_magnitute > threshold:
                    img[y][x] = 255
            #         img[first_neighb_y][first_neighb_x] = 0
            #         img[second_neighb_y][second_neighb_x] = 0
            #         grad_magnitude[first_neighb_y][first_neighb_x] = 0
            #         grad_magnitude[second_neighb_y][second_neighb_x] = 0
            # else:
            #     img[y][x] = 0
            #     grad_magnitude[y][x] = 0

    return img, grad_magnitude


input_img_path = os.path.join(input_dir, img_name)
output_dir_path = os.path.join(output_dir_name, img_name)

img = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)
print('img.shape = {}'.format(img.shape))

blurred_img = cv2.GaussianBlur(img, (5, 5), 1.4)
cv2.imwrite(os.path.join(output_dir_path, 'gaussian_blur.png'), blurred_img)

sobel_x = cv2.Sobel(blurred_img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred_img, cv2.CV_64F, 0, 1, ksize=3)

cv2.imwrite(os.path.join(output_dir_path, 'sobel_x.png'), sobel_x)
cv2.imwrite(os.path.join(output_dir_path, 'sobel_y.png'), sobel_y)

grad_magnitude = np.sqrt(np.add(np.power(sobel_x, 2), np.power(sobel_y, 2)))
cv2.imwrite(os.path.join(output_dir_path, 'grad_magnitude.png'), grad_magnitude)
grad_magnitude_img = cv2.imread(os.path.join(output_dir_path, 'grad_magnitude.png'), cv2.IMREAD_GRAYSCALE)
print('grad_magnitude_img max = {}'.format(np.amax(grad_magnitude_img)))

grad_direction = np.arctan2(sobel_y, sobel_x)
grad_direction = grad_direction * 180 / np.pi

print('grad_magnitude.shape = {}'.format(grad_magnitude.shape))
print('sobel_x max = {}'.format(np.amax(sobel_x)))
print('grad_magnitude max = {}'.format(np.amax(grad_magnitude)))
print('grad_magnitude min = {}'.format(np.amin(grad_magnitude)))
# print('grad_direction = {}'.format(grad_direction))

non_maximum_suppressed_img, grad_magnitude = non_maximum_suppression(grad_magnitude_img, grad_magnitude, grad_direction)
non_maximum_suppressed_img[non_maximum_suppressed_img < 255] = 0

cv2.imwrite(os.path.join(output_dir_path, 'non_maximum_suppressed_img.png'), non_maximum_suppressed_img)

thresholded_with_hysterysis_img, is_changed = threshold_with_hysterysis(non_maximum_suppressed_img, grad_direction, grad_magnitude)

i = 0
while is_changed:
    thresholded_with_hysterysis_img, is_changed = threshold_with_hysterysis(thresholded_with_hysterysis_img,
                                                                            grad_direction,
                                                                            grad_magnitude)

    if i % 10 == 0:
        cv2.imwrite(os.path.join(output_dir_path, 'thresholded_with_hysterysis_img_{}.png'.format(i)),
                    thresholded_with_hysterysis_img)

    i = i + 1
    print('Thresholded with Hysterysis: {}'.format(i))

if not os.path.exists(output_dir_path):
    os.makedirs(output_dir_path)

cv2.imwrite(os.path.join(output_dir_path, 'thresholded_with_hysterysis_img_final.png'.format(i)),
                    thresholded_with_hysterysis_img)
