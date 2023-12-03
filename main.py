import sys

import numpy
import numpy as np
import math
import cv2 as cv
import os
from skimage.morphology import skeletonize, thin
from skimage.util import invert


def average_grayscale_conversion(image):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i][j][0] = image[i][j][0]/3 + image[i][j][1]/3 + image[i][j][2]/3
    image = image[:, :, 0]
    return image


def weighted_grayscale_conversion(image):
    # Grayscale = 0.299R + 0.587G + 0.114B
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i][j][0] = np.uint8(0.288 * image[i][j][2] + 0.587 * image[i][j][1] + 0.114 * image[i][j][2])
    image = image[:, :, 0]
    return image


def normalization_grayscale(image):
    img_max = image.max()
    img_min = image.min()
    if not img_max == 255 and img_min == 0:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[i][j] = 255 * (image[i][j] - img_min) / (img_max - img_min)
    return image


def shear(image, x_shear, y_shear):
    max_y_shear = round(abs(image.shape[1] * y_shear))
    max_x_shear = round(abs(image.shape[0] * x_shear))
    new_height = image.shape[0] + max_y_shear
    new_width = image.shape[1] + max_x_shear
    temp = np.zeros(shape=(new_height, new_width))
    for i in range(image.shape[0]):
        if x_shear < 0:
            shift_x = round(i * x_shear + max_x_shear)
        elif x_shear > 0:
            shift_x = round(i * x_shear)
        else:
            shift_x = 0
        for j in range(image.shape[1]):
            if y_shear < 0:
                shift_y = round(j * y_shear) + max_y_shear
            elif y_shear > 0:
                shift_y = round(j * y_shear)
            else:
                shift_y = 0
            temp[i + shift_y][j + shift_x] = image[i][j]
    return temp


def rotate(image, angle):
    angle = math.radians(angle)
    shear_x = -math.tan(angle/2)
    shear_y = math.sin(angle)
    image = shear(image, shear_x, 0)
    image = shear(image, 0, shear_y)
    image = shear(image, shear_x, 0)
    image = image[~np.all(image == 0, axis=1)]
    image = image[:, ~np.all(image == 0, axis=0)]
    return image


def binarization(image, threshold):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] > threshold:
                image[i][j] = 255
            else:
                image[i][j] = 0
    return image


def projection_profile_skew(image, max_skew):
    sum_in_row = np.zeros(2*max_skew+1, image.shape[1])
    variation = np.zeros(2*max_skew+1)
    for skew in range(-max_skew, max_skew):
        image = rotate(image, 1)
        for row, i in image:
            sum_in_row[skew][i] = row.sum()
        variation[skew] = np.var(sum_in_row[skew])

    return image


def thresholding_otsu(image):
    image = normalization_grayscale(image)
    _, image = cv.threshold(image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    return image


def adaptive_thresholding(image):
    image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 251, 20)
    return image


def denoising(image):
    image = cv.fastNlMeansDenoising(src=image, h=30, templateWindowSize=7, searchWindowSize=21)
    return image

def preprocessing_folder(folder_path):
    for image in os.listdir(folder_path):
        preprocessing(folder_path + image)


def show_image(img):
    cv.imshow('img', img)
    cv.waitKey()


def preprocessing(image_path):
    img = cv.imread(image_path)
    show_image(img)
    img = weighted_grayscale_conversion(img)
    show_image(img)
    img = adaptive_thresholding(img)
    show_image(img)
    # img = projection_profile_skew(img, 15)
    img = invert(img)
    img = skeletonize(img).astype(np.uint8)
    img = normalization_grayscale(img)
    show_image(img)
    img = denoising(img)
    show_image(img)
    if not os.path.exists('PreprocessedImages/'):
        os.makedirs('PreprocessedImages/')
    image_name = os.path.basename(os.path.normpath(image_path))
    cv.imwrite('PreprocessedImages/' + image_name, img)


preprocessing('HandText.jpg')

