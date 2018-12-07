import os
import cv2


def save_image(output_dir_path, img, filename, jpeg_quality = 100):
    output_img_path = os.path.join(output_dir_path, filename)

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    success = cv2.imwrite(output_img_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])

    if success:
        return output_img_path
    else:
        raise Exception('[ERROR] A result image was not saved!')