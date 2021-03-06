import os
import cv2
import numpy as np
import pandas as pd
import math

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from skimage.util import random_noise


from common import utils

IMAGES = ['war.jpg', 'house.jpg', 'boat.jpg']
# IMAGES = ['war.jpg']
IMG_NAME = 'penguins.jpg'
INPUT_DIR = 'sift' + os.sep + 'input'
OUTPUT_DIR = 'sift' + os.sep + 'output'
TEMP_FOLDER = OUTPUT_DIR + os.sep + IMG_NAME + os.sep + 'temp'
EXTRACT_FEATURES_MODES = ['SIFT', 'BRIEF', 'Harris', 'SURF', 'ORB']
TRANSFORMATION_TYPE = 'scale'
# EXTRACT_FEATURES_MODES = ['SURF']
N_KEYPOINTS = 100
VARIANCE_VALUES = [5, 10, 20, 40, 100]
SCALE_VALUES = [0.5, 0.25, 0.125, 0.0625]
ROTATE_VALUES = range(0, 370, 10)
QUALITY_VALUES = range(5, 105, 5)
FILTER_POINTS_THRESHOLD = 10
JPEG_IMG_QUALITY = [int(cv2.IMWRITE_JPEG_QUALITY), 100]


def add_noise(img, variance):
    return (random_noise(img, var=variance) * 255).astype(np.uint8)


def sort_keypoints(kps, vector_size):
    return sorted(kps, key=lambda x: -x.response)[:vector_size]


def extract_features(img, vector_size, mode):
    allowed_modes = ['SIFT', 'Harris', 'SURF', 'BRIEF', 'ORB']

    if mode not in allowed_modes:
        raise ValueError('mode = {} is not valid feature extraction mode'.format(mode))

    if mode == 'SIFT':
        sift = cv2.xfeatures2d.SIFT_create()

        kps = sift.detect(img, None)
        kps = sort_keypoints(kps, vector_size)

        kps, descriptors = sift.compute(img, kps)

    elif mode == 'Harris':
        harris_fd = cv2.xfeatures2d.HarrisLaplaceFeatureDetector_create()
        kps = harris_fd.detect(img, None)
        kps = sort_keypoints(kps, vector_size)
        sift = cv2.xfeatures2d.SIFT_create()
        kps, descriptors = sift.compute(img, kps)

    elif mode == 'SURF':
        surf = cv2.xfeatures2d.SURF_create()
        kps = surf.detect(img, None)
        kps = sort_keypoints(kps, vector_size)
        kps, descriptors = surf.compute(img, kps)

    elif mode == 'BRIEF':
        star = cv2.xfeatures2d.StarDetector_create()
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

        kps = star.detect(img, None)
        kps = sort_keypoints(kps, vector_size)
        kps, descriptors = brief.compute(img, kps)

    elif mode == 'ORB':
        orb = cv2.ORB_create()

        kps = orb.detect(img, None)
        kps = sort_keypoints(kps, vector_size)
        kps, descriptors = orb.compute(img, kps)

    return kps, descriptors


def produce_transformed_image(img, mode):
    allowed_modes = ['noise', 'scale', 'rotate', 'quality']

    if mode not in allowed_modes:
        raise ValueError('mode = {} is not valid transformation mode'.format(mode))

    if mode == 'noise':
        for variance in VARIANCE_VALUES:
            noisy_img = add_noise(img, variance / 255.0)
            yield noisy_img, variance

    elif mode == 'scale':
        for scale in SCALE_VALUES:
            resized_img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            yield resized_img, scale

    elif mode == 'rotate':
        (h, w) = img.shape
        center = (w // 2, h // 2)

        for angle in ROTATE_VALUES:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_img = cv2.warpAffine(img, M, (w, h))
            yield rotated_img, angle

    elif mode == 'quality':
        for quality in QUALITY_VALUES:
            jpeg_quality = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

            if not os.path.exists(TEMP_FOLDER):
                os.makedirs(TEMP_FOLDER)

            filename = 'quality={}.jpg'.format(quality)
            output_img_path = os.path.join(TEMP_FOLDER, filename)
            success = cv2.imwrite(output_img_path, img, jpeg_quality)

            if not success:
                raise Exception('File cannot be written: {}'.format(output_img_path))

            img = cv2.imread(output_img_path, cv2.IMREAD_GRAYSCALE)

            os.remove(output_img_path)

            yield img, quality


def get_img_center_point(img_shape):
    return (img_shape[1] // 2, img_shape[0] // 2)


def filter_matches_rotate(matches, query_kps, train_kps, angle, rotation_point):
    result_matches = []

    angle = math.radians(angle)

    for match in matches:
        x = query_kps[match.queryIdx].pt[0] - rotation_point[0]
        y = query_kps[match.queryIdx].pt[1] - rotation_point[1]

        x_rotated = x * math.cos(angle) + y * math.sin(angle)
        y_rotated = y * math.cos(angle) - x * math.sin(angle)

        x_rotated = x_rotated + rotation_point[0]
        y_rotated = y_rotated + rotation_point[1]

        diff_x = abs(x_rotated - train_kps[match.trainIdx].pt[0])
        diff_y = abs(y_rotated - train_kps[match.trainIdx].pt[1])

        if diff_x < FILTER_POINTS_THRESHOLD and diff_y < FILTER_POINTS_THRESHOLD:
            result_matches.append(match)

    return result_matches


def filter_matches(matches, query_kps, train_kps, transformation_mode, angle = 0, img_shape = None, scale = 1):
    result_matches = []

    if transformation_mode == 'quality' or transformation_mode == 'noise':
        for match in matches:
            x = query_kps[match.queryIdx].pt[0]
            y = query_kps[match.queryIdx].pt[1]

            diff_x = abs(x - train_kps[match.trainIdx].pt[0])
            diff_y = abs(y - train_kps[match.trainIdx].pt[1])

            if diff_x < FILTER_POINTS_THRESHOLD and diff_y < FILTER_POINTS_THRESHOLD:
                result_matches.append(match)

    elif transformation_mode == 'rotate':
        center_point = get_img_center_point(img_shape)
        result = filter_matches_rotate(matches, query_kps, train_kps, angle, center_point)

        return result

    elif transformation_mode == 'scale':
        for match in matches:
            x = query_kps[match.queryIdx].pt[0] * scale
            y = query_kps[match.queryIdx].pt[1] * scale

            diff_x = abs(x - train_kps[match.trainIdx].pt[0])
            diff_y = abs(y - train_kps[match.trainIdx].pt[1])

            if diff_x < FILTER_POINTS_THRESHOLD and diff_y < FILTER_POINTS_THRESHOLD:
                result_matches.append(match)
        return result_matches

    else:
        raise ValueError()

    return result_matches


def count_matches_features(input_img_path, output_img_dir, mode):
    img_origin = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)

    kps_origin, descriptors_origin = extract_features(img_origin, N_KEYPOINTS, mode)

    # origin_img_with_kps = cv2.drawKeypoints(img_origin, kps_origin, None,
    #                                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # utils.save_image(output_img_dir, origin_img_with_kps, 'origin_keypoints.jpg')

    matches_counts = []
    transform_feature_values = []

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    for transformed_img, transform_feature_value in produce_transformed_image(img_origin, TRANSFORMATION_TYPE):
        kps, descriptors = extract_features(transformed_img, N_KEYPOINTS, mode)
        transformed_img_with_kps = cv2.drawKeypoints(transformed_img, kps, None,
                                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        filename = '{}={}.jpg'.format(TRANSFORMATION_TYPE, transform_feature_value)
        utils.save_image(output_img_dir, transformed_img_with_kps, filename)

        if descriptors is not None:
            if descriptors_origin.shape[0] < descriptors.shape[0]:
                np.append(descriptors_origin, np.zeros((descriptors.shape[0] - descriptors_origin.shape[0], descriptors_origin.shape[1])), axis=0)
            elif descriptors_origin.shape[0] > descriptors.shape[0]:
                np.append(descriptors, np.zeros((descriptors_origin.shape[0] - descriptors.shape[0], descriptors_origin.shape[1])), axis=0)

            matches = bf.match(descriptors_origin, descriptors)

            print('{}. {} = {}. Matches = {}'.format(input_img_path, TRANSFORMATION_TYPE, transform_feature_value, len(matches)))
            matches = filter_matches(matches, kps_origin, kps, TRANSFORMATION_TYPE, transform_feature_value, img_origin.shape, transform_feature_value)
            print('Filtered matches = {}'.format(len(matches)))

            matches_filename = 'matches_{}={}.jpg'.format(TRANSFORMATION_TYPE, transform_feature_value)

            if TRANSFORMATION_TYPE == 'scale':
                img_to_compare = transformed_img
            else:
                img_to_compare = img_origin

            img_matches = cv2.drawMatches(img_origin, kps_origin, img_to_compare, kps, matches, None, flags=2)
            utils.save_image(output_img_dir, img_matches, matches_filename)
        else:
            matches = []

        matches_count = len(matches)
        matches_counts.append(matches_count)
        transform_feature_values.append(transform_feature_value)

    # print('{} = {}. Matches = {}'.format(TRANSFORMATION_TYPE, transform_feature_value, matches_count))
    return matches_counts, transform_feature_values


def count_matches(input_img_path, output_dir, mode):
    output_dir_method = os.path.join(output_dir, mode)
    matches_counts, transform_feature_values = count_matches_features(input_img_path, output_dir_method, mode)

    return matches_counts, transform_feature_values


def main():
    chart_data = pd.DataFrame()

    for index, features_extract_method in enumerate(EXTRACT_FEATURES_MODES):
        matches_counts_images = []

        print('Working on: {}/{} - {}'.format(index + 1, len(EXTRACT_FEATURES_MODES), features_extract_method))

        for img_filename in IMAGES:
            print(img_filename)
            input_img_path = os.path.join(INPUT_DIR, img_filename)
            output_dir = os.path.join(OUTPUT_DIR, img_filename)

            matches_counts, transform_feature_values = count_matches(input_img_path, output_dir, features_extract_method)
            matches_counts_images.append(matches_counts)

        matches_counts_images = np.array(matches_counts_images)

        matches_counts_average = np.sum(matches_counts_images, axis=0) / float(matches_counts_images.shape[0])

        print(matches_counts_images)
        print(matches_counts_average)

        line_data = pd.DataFrame({'x': transform_feature_values, features_extract_method: matches_counts_average})
        chart_data = chart_data.append(line_data, sort=False)

        plt.plot('x', features_extract_method, data=chart_data)

    plt.title('Dependency between {} and number of matching features'.format(TRANSFORMATION_TYPE))
    plt.xlabel('Transformation value')
    plt.ylabel('Number of matching features')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    chart_filename = 'chart_{}.png'.format(TRANSFORMATION_TYPE)
    chart_filepath = os.path.join(OUTPUT_DIR, chart_filename)
    plt.savefig(chart_filepath, bbox_inches='tight')


if __name__ == '__main__':
    main()
