import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

input_dir = 'input'
output_dir_name = 'output'
img_name = "lena.png"
window_size_of_maximum_suppression = (3,3)


def non_maximum_suppression(img, grad_magnitude, grad_direction):
    for y in range(0, img.shape[0], 1):
        for x in range(0, img.shape[1], 1):
            current_magnitute = grad_magnitude[y][x]
            current_direction = grad_direction[y][x]

            if 22.5 <= current_direction < 67.5:
                first_neighb_x = x + 1
                first_neighb_y = y - 1
                second_neighb_x = x - 1
                second_neighb_y = y + 1
            elif 67.5 <= current_direction < 112.5:
                first_neighb_x = x + 1
                first_neighb_y = y
                second_neighb_x = x - 1
                second_neighb_y = y
            elif 112.5 <= current_direction < 157.5:
                first_neighb_x = x - 1
                first_neighb_y = y - 1
                second_neighb_x = x + 1
                second_neighb_y = y + 1
            elif 157.5 <= current_direction <= 180 or 0 <= current_direction < 22.5:
                first_neighb_x = x
                first_neighb_y = y - 1
                second_neighb_x = x
                second_neighb_y = y + 1

            if first_neighb_x >= img.shape[1] or first_neighb_y >= img.shape[0]:
                first_neighb_magnitude = 0
            else:
                first_neighb_magnitude = grad_magnitude[first_neighb_y][first_neighb_x]

            if second_neighb_x >= img.shape[1] or second_neighb_y >= img.shape[0]:
                second_neighb_magnitude = 0
            else:
                second_neighb_magnitude = grad_magnitude[second_neighb_y][second_neighb_x]

            if current_magnitute > first_neighb_magnitude and current_magnitute > second_neighb_magnitude:
                if first_neighb_x < img.shape[1] and first_neighb_y < img.shape[0]:
                    img[first_neighb_y][first_neighb_x] = 0
                if second_neighb_x < img.shape[1] and second_neighb_y < img.shape[0]:
                    img[second_neighb_y][second_neighb_x] = 0

                img[y][x] = 255

    return img


input_img_path = os.path.join(input_dir, img_name)
output_dir_path = os.path.join(output_dir_name, img_name)

img = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)
print('img.shape = {}'.format(img.shape))

blurred_img = cv2.GaussianBlur(img, (5, 5), 1.4)
sobel_x = cv2.Sobel(blurred_img, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(blurred_img, cv2.CV_64F, 0, 1, ksize=5)

grad_magnitude = np.sqrt(np.add(np.power(sobel_x, 2), np.power(sobel_y, 2)))
grad_direction = np.arctan2(sobel_y, sobel_x)
grad_direction = grad_direction * 180 / np.pi

print('grad_magnitude.shape = {}'.format(grad_magnitude.shape))
# print('grad_magnitude = {}'.format(grad_magnitude))
# print('grad_direction = {}'.format(grad_direction))

supressed_img = non_maximum_suppression(img, grad_magnitude, grad_direction)

if not os.path.exists(output_dir_path):
    os.makedirs(output_dir_path)

cv2.imwrite(os.path.join(output_dir_path, 'gaussian_blur.png'), blurred_img)
cv2.imwrite(os.path.join(output_dir_path, 'sobel_x.png'), sobel_x)
cv2.imwrite(os.path.join(output_dir_path, 'sobel_y.png'), sobel_y)
cv2.imwrite(os.path.join(output_dir_path, 'supressed_img.png'), supressed_img)

