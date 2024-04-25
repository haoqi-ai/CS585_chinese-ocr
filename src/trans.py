import os
import sys
import argparse
from pathlib import Path
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from src.helper import filter_component


def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def colorize(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def binarize(image):
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 15
    )


def expand(image, height, width):
    expand_height = max(int((height - image.shape[0]) // 2), 0)
    expand_width = max(int((width - image.shape[1]) // 2), 0)
    return cv2.copyMakeBorder(
        image,
        expand_height,
        expand_height,
        expand_width,
        expand_width,
        cv2.BORDER_CONSTANT,
    )


def rotate(image, angle):
    # expand
    expand_height = image.shape[0] * math.cos(math.radians(abs(angle))) + image.shape[
        1
    ] * math.sin(math.radians(abs(angle)))
    expand_width = image.shape[0] * math.sin(math.radians(abs(angle))) + image.shape[
        1
    ] * math.cos(math.radians(abs(angle)))
    image = expand(image, expand_height, expand_width)

    # rotate
    rotate_matrix = cv2.getRotationMatrix2D(
        (image.shape[1] / 2, image.shape[0] / 2), angle, 1
    )
    image_rotate = cv2.warpAffine(
        image, rotate_matrix, (image.shape[1], image.shape[0])
    )

    return image_rotate


def find_rotation(image):
    image = to_grayscale(image)

    # binarize
    image = binarize(image)

    # test
    score = []
    for i in np.arange(-5, 5, 0.1):
        # rotate
        image_rotate = rotate(image, i)

        # calc score
        col_average = cv2.reduce(image_rotate, dim=0, rtype=cv2.REDUCE_AVG).flatten()
        row_average = cv2.reduce(image_rotate, dim=1, rtype=cv2.REDUCE_AVG).flatten()
        score.append(("vertical", i, col_average.std()))
        score.append(("horizontal", i, row_average.std()))

    # find best angle
    score.sort(key=lambda x: x[2], reverse=True)

    return score[0][:2]


def calc_fillrate(image, filter_mode="min"):
    row_fillrate = cv2.reduce(image, dim=1, rtype=cv2.REDUCE_AVG).flatten() / 255
    col_fillrate = cv2.reduce(image, dim=0, rtype=cv2.REDUCE_AVG).flatten() / 255

    if filter_mode == "min":
        from scipy import ndimage

        col_fillrate = ndimage.minimum_filter(
            col_fillrate, size=int(image.shape[1] / 100), mode="nearest"
        )

    elif filter_mode == "median":
        from scipy.signal import medfilt

        col_fillrate = medfilt(
            col_fillrate,
            kernel_size=int(image.shape[1] / 50) + 0
            if int(image.shape[1] / 50) % 2
            else 1,
        )

    elif filter_mode == "conv":
        from scipy.signal import convolve

        window_size = int(image.shape[1] / 100)
        col_fillrate = convolve(
            col_fillrate, np.full(window_size, 1 / window_size), mode="same"
        )

    return row_fillrate, col_fillrate


def execute_split(img, rows, cols):
    imgs = []
    # for row_idx in range(len(rows) - 1):
    #     row_img = img[rows[row_idx] : rows[row_idx + 1], :]
    #     for col_idx in range(len(cols) - 1):
    #         word_img = row_img[:, cols[col_idx] : cols[col_idx + 1]]
    #         imgs.append(word_img)
    cols = np.insert(cols, 0, 0)
    cols = np.append(cols, img.shape[1])
    rows = np.insert(rows, 0, 0)
    rows = np.append(rows, img.shape[0])
    for col_idx in range(len(cols) - 2, -1, -1):
        col_img = img[:, cols[col_idx] : cols[col_idx + 1]]
        for row_idx in range(len(rows) - 1):
            word_img = col_img[rows[row_idx] : rows[row_idx + 1], :]
            imgs.append(word_img)
    return imgs


def split_on_fillrate(image):
    # calc fillrate
    row_fillrate, col_fillrate = calc_fillrate(image)

    # find gaps
    row_gaps = find_peaks(row_fillrate * -1, distance=int(image.shape[0] / 30))[0]
    # print('row gaps:', row_gaps)
    col_gaps = find_peaks(
        col_fillrate * -1,
        distance=int(image.shape[1] / 30),
        width=int(image.shape[1] / 200),
    )[0]
    # print('col gaps:', col_gaps)

    # filter gaps
    col_gap_fillrate = col_fillrate[col_gaps]
    col_gaps = col_gaps[col_gap_fillrate - col_gap_fillrate.mean() < 0.05]
    # print(col_gap_fillrate)

    # split image
    image_split = colorize(image)
    image_split = execute_split(image_split, row_gaps, col_gaps)
    # for row in row_gaps:
    #     image_split[row-1:row+2, 0:image_split.shape[1]] = [0, 255, 0]

    # for col in col_gaps:
    #     image_split[0:image_split.shape[0], col-1:col+2] = [0, 0, 255]

    return image_split


def remove_outline(img):
    for row in range(img.shape[0]):
        if sum(img[row]) < 10000 and (row < 5 or row > img.shape[0] - 5):
            img[row] = 255

    # img[:, img.shape[1]-20:] = 255
    # img[:, :20] = 255
    # img[:20] = 255
    # img[img.shape[0]-20:] = 255
    return img


def transform(image):
    image = remove_outline(image)
    
    orientation, rotation_angle = find_rotation(image)
    image_rotate = rotate(image, rotation_angle)

    image_gray = to_grayscale(image_rotate)

    image_binary = binarize(image_gray)
    # plt.imshow(image_binary)
    # plt.show()
    image_binary = filter_component(image_binary)
    # plt.imshow(image_binary)
    # plt.show()
    image = image_binary

    # split
    # image_split = split_on_fillrate(image_binary)

    return image

