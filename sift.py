import os
import cv2
import math
import numpy as np
from skimage.util import random_noise
# from matplotlib import pyplot as plt

from common import utils

INPUT_DIR = 'sift' + os.sep + 'input'
OUTPUT_DIR = 'sift' + os.sep + 'output'
IMG_NAME = 'penguins.jpg'
VARIANCE_VALUES = [0.05, 0.1, 0.2, 0.4, 1]
N_KEYPOINTS = 100
JPEG_IMG_QUALITY = [int(cv2.IMWRITE_JPEG_QUALITY), 100]


def add_noise(img, variance):
    return (random_noise(img, var=variance) * 255).astype(np.uint8)


def extract_features(img, vector_size):
    sift = cv2.xfeatures2d.SIFT_create()

    kps = sift.detect(img, None)
    kps = sorted(kps, key=lambda x: -x.response)[:vector_size]

    kps, descriptors = sift.compute(img, kps)

    return kps, descriptors


def main():
    input_img_path = os.path.join(INPUT_DIR, IMG_NAME)
    output_img_dir = os.path.join(OUTPUT_DIR, IMG_NAME)

    origin_img = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)

    kps, descriptors = extract_features(origin_img, N_KEYPOINTS)

    origin_img_with_kps = cv2.drawKeypoints(origin_img, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    utils.save_image(output_img_dir, origin_img_with_kps, 'sift_keypoints.jpg', 100)

    for variance in VARIANCE_VALUES:
        noisy_img = add_noise(origin_img, variance)
        filename = 'img_with_noise_var={}.jpg'.format(variance)

        noisy_kps, noisy_descriptors = extract_features(noisy_img, N_KEYPOINTS)

        noisy_img_with_kps = cv2.drawKeypoints(noisy_img, noisy_kps, None,
                                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        utils.save_image(output_img_dir, noisy_img_with_kps, filename, 100)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        matches = bf.match(descriptors, noisy_descriptors)

        matches = sorted(matches, key=lambda x: x.distance)

        matches_filename = 'matches_var={}.jpg'.format(variance)
        img_matches = cv2.drawMatches(origin_img, kps, origin_img, noisy_kps, matches, None, flags=2)
        utils.save_image(output_img_dir, img_matches, matches_filename, 100)

        # blank_img = np.zeros(noisy_img.shape, dtype=np.uint8)
        # test = cv2.drawKeypoints(blank_img, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # test2 = cv2.drawKeypoints(blank_img, noisy_kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # utils.save_image(output_img_dir, test, 'test.jpg', 100)
        # utils.save_image(output_img_dir, test2, 'test2.jpg', 100)
        # break

        print(
            'Found {} kps in noisy img with variance = {}. Matches = {}'.format(len(noisy_kps), variance, len(matches)))


if __name__ == '__main__':
    main()
