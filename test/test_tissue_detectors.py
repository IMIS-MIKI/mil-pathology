# Test different automatic tissue detectors from https://imagej.net/plugins/auto-threshold#li
# Methods to try: Otsu, Huang, Mean, Min Error(1), Percentil, Triangle

# Results
# Min_Error 1 prone to not converging, so removed it
# Percentil value highly dependent on the ratio of image to background, so also not ideal
# Median same as percentil 50
# Mean overall has worst performance than the remaining

# Best ones are Triangle, Otsu and Huang
# Otsu tends to underestimate the tissue area (in just a couple of cases is it better than the Triangle)
# Huang tends to overestimate the tissue area

# Overall Triangle seems to have the best performance in detecting the tissue sections

import os
import shutil
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from matplotlib import pyplot as plt
from math import fabs, log, log10, sqrt, floor, ceil

# preprocessing parameters
input_path = Path("data/png_files")
output_path = Path("data/test/test_tissue_detector")

name_csv = 'data_preprocessing_kongo.csv'
image_type = "fluorescent"

##########################


class PreProcessing:
    """ Description for pre-processing class """

    # path containing category folders
    input_path = Path("input")

    # Folder path to store the output, after slicing the images
    output_path = Path("output")

    # x, y dimensions of each slice
    tile_size = [224, 224]

    # Shift applied to the slices when extracting crop (for data augmentation) - Instead of starting on pixel 0, starts
    # on the one identified by the shift
    tile_shift = [0, 0]

    # Step applied to the slices when extracting crop (for data augmentation) - Controls the overlapping between
    # consecutive slices (positive number leads to overlap)
    tile_step = [0, 0]

    # Required percentage of tissue, in a image, to be classified as valid for the classification process
    valid_background_perc = 0.9

    # Name of the csv file to be produced (where all the information regarding patients, path for the slices and
    # classification is stored)
    name_csv = 'image-data.csv'

    # Ratio to split training and test set
    split_ratio = 0.7

    def __init__(self, input_path=None, output_path=None, tile_size=None, tile_shift=None, tile_step=None,
                 name_csv=None, valid_background_perc=None, split_ratio=None, verbose=False, image_type=None):
        # Validate classes inputs
        if input_path is not None:
            self.input_path = input_path
        if output_path is not None:
            self.output_path = output_path
        if tile_size is not None:
            self.tile_size = tile_size
        if tile_shift is not None:
            self.tile_shift = tile_shift
        if tile_step is not None:
            self.tile_step = tile_step
        if name_csv is not None:
            self.name_csv = name_csv
        if valid_background_perc is not None:
            self.valid_background_perc = valid_background_perc
        if split_ratio is not None:
            self.split_ratio = split_ratio
        if image_type is not None:
            self.image_type = image_type

        self.nb_input_channels = 0

        self.verbose = verbose
        self.train_set_statistics = (None, None)

        if self.verbose:
            print("Pre-processing initialized successfully")

    def channel_selection(self, img, channel=2, color_convert=None):
        # Full rgb image
        if self.image_type == "normal":
            # Convert to HSV color space
            color_convert = cv2.COLOR_BGR2HSV
            channel = 2

        # Image already stored as HSV
        elif self.image_type == "fluorescent":
            color_convert = None
            channel = 2

        if color_convert:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        return img[:, :, channel]


def linear_bias_removal(img):
    """Estimates vertical and horizontal linear pattern and removes it from the image
    Currently only removing slope (Example C from remove_bias_preprocessing)"""

    # Remove linear Horizontal influence
    model = np.polyfit(range(img.shape[1]), np.median(img, 0), 1)

    for i in range(img.shape[1]):
        new_array = img[:, i] - (i * model[0])  # + model[1])
        new_array[new_array < 0] = 0
        img[:, i] = new_array

    # Remove linear vertical influence
    model = np.polyfit(range(img.shape[0]), np.median(img, 1), 1)

    for i in range(img.shape[0]):
        new_array = img[i, :] - (i * model[0])  # + model[1])
        new_array[new_array < 0] = 0
        img[i, :] = new_array

    # cv2.imwrite("data/tmp_both2.png", img)
    return img

def get_tissue_detector_mask(img, method="otsu"):
    """ Apply defined tissue detector technique. Based on threshold detectors
    slice_path: string with path to the image
    gaussian: blur parameter in pixels, None for no blur
    invert_list: list of channels (same shape as channel_list) to invert, None for no inversion
    """

    if method == "otsu":
        return get_otsu_tissue_detector_mask(img)

    if method == "triangle":
        return get_triangle_tissue_detector_mask(img)

    if method == "huang":
        return get_huang_tissue_detector_mask(img)

    if method == "mean":
        return get_mean_tissue_detector_mask(img)

    if method == "median":
        return get_median_tissue_detector_mask(img)

    if method == "min_error_1":
        return get_min_error_1_tissue_detector_mask(img)

    if method.startswith("percentil_"):
        ratio = int(method.split("_")[-1]) / 100.

        return get_percentil_tissue_detector_mask(img, ratio)


def get_otsu_tissue_detector_mask(img):
    """ Apply the Otsu technique, in order to differentiate tissue from background.
    Returns a mask where True/1 means background and False/0 means tissue.
    img:
    """
    # "Inspired" from here: https://docs.opencv.org/4.5.1/d7/d4d/tutorial_py_thresholding.html
    # http://web-ext.u-aizu.ac.jp/course/bmclass/documents/otsu1979.pdf

    # Otsu thresholding and mask generation
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return mask


def get_triangle_tissue_detector_mask(img):
    """ Apply the triangle threshold method.
    https://pythonmana.com/2021/10/20211009194205124c.html
    img:
    """

    # Triangle thresholding and mask generation
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

    return mask


def get_huang_tissue_detector_mask(img, verbose=False):
    """ Apply the triangle threshold method.
    img:
    """

    # https://github.com/dnhkng/Huang-Thresholding/blob/master/Threshold.py
    def get_huang_threshold(data):
        """Implements Huang's fuzzy thresholding method
            Uses Shannon's entropy function (one can also use Yager's entropy function)
            Huang L.-K. and Wang M.-J.J. (1995) "Image Thresholding by Minimizing
            the Measures of Fuzziness" Pattern Recognition, 28(1): 41-51"""

        threshold = -1

        first_bin = 0
        for ih in range(254):
            if data[ih] != 0:
                first_bin = ih
                break

        last_bin = 254
        for ih in range(254, -1, -1):
            if data[ih] != 0:
                last_bin = ih
                break

        if first_bin == last_bin:
            return 0

        term = 1.0 / (last_bin - first_bin)

        if verbose:
            print(first_bin, last_bin, term)
        mu_0 = np.zeros(shape=(254, 1))
        num_pix = 0.0
        sum_pix = 0.0
        for ih in range(first_bin, 254):
            sum_pix = sum_pix + (ih * data[ih])
            num_pix = num_pix + data[ih]
            mu_0[ih] = sum_pix / num_pix  # NUM_PIX cannot be zero !

        mu_1 = np.zeros(shape=(254, 1))
        num_pix = 0.0
        sum_pix = 0.0
        for ih in range(last_bin, 1, -1):
            sum_pix = sum_pix + (ih * data[ih])
            num_pix = num_pix + data[ih]

            mu_1[ih - 1] = sum_pix / num_pix  # NUM_PIX cannot be zero !

        min_ent = float("inf")
        for it in range(254):
            ent = 0.0
            for ih in range(it):
                # Equation (4) in Reference
                mu_x = 1.0 / (1.0 + term * fabs(ih - mu_0[it]))
                if not (mu_x < 1e-06 or mu_x > 0.999999):
                    # Equation (6) & (8) in Reference
                    ent = ent + data[ih] * (-mu_x * log(mu_x) - (1.0 - mu_x) * log(1.0 - mu_x))

            for ih in range(it + 1, 254):
                # Equation (4) in Ref. 1 */
                mu_x = 1.0 / (1.0 + term * fabs(ih - mu_1[it]))
                if not (mu_x < 1e-06 or mu_x > 0.999999):
                    # Equation (6) & (8) in Reference
                    ent = ent + data[ih] * (-mu_x * log(mu_x) - (1.0 - mu_x) * log(1.0 - mu_x))
            if ent < min_ent:
                min_ent = ent
                threshold = it

            if verbose:
                print("min_ent, threshold ", min_ent, threshold)

        return threshold

    # Create histogram from image
    hist = np.histogram(img, bins=np.arange(256))[0]

    # Calculate threshold
    threshold = get_huang_threshold(hist)

    # Use found threshold and get the mask
    _, mask = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    return mask


def get_mean_tissue_detector_mask(img):
    """ Apply the mean threshold method.
    img:
    """

    # Calculate threshold
    threshold = np.mean(img)

    # Use found threshold and get the mask
    _, mask = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    return mask


def get_median_tissue_detector_mask(img):
    """ Apply the median threshold method.
    img:
    """

    # Calculate threshold
    threshold = np.median(img)

    # Use found threshold and get the mask
    _, mask = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    return mask


def get_percentil_tissue_detector_mask(img, ratio):
    """ Apply the percentil threshold method.
    img:
    """

    # https://imagej.nih.gov/ij/download/tools/source/ij/process/AutoThresholder.java
    def get_percentil_threshold(data, ratio):

        threshold = -1
        ptile = ratio  # default fraction of foreground pixels
        total = np.sum(data)
        temp = 1.0
        cumsum = 0

        for i in range(255):
            cumsum = cumsum + data[i]
            value = abs((cumsum / total) - ptile)
            if value < temp:
                temp = value
                threshold = i

        return threshold

    # Create histogram from image
    hist = np.histogram(img, bins=np.arange(256))[0]

    # Calculate threshold
    threshold = get_percentil_threshold(hist, ratio)

    # Use found threshold and get the mask
    _, mask = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    return mask


def get_min_error_1_tissue_detector_mask(img):
    """ Apply the minError(1) threshold method.
    img:
    """

    # https://imagej.nih.gov/ij/download/tools/source/ij/process/AutoThresholder.java
    def get_min_error_threshold(data):

        def A(y, j):
            x = 0
            j = int(j)
            for i in range(j):
                x = x + y[i]
            return x

        def B(y, j):
            x = 0
            j = int(j)
            for i in range(j):
                x = x + i * y[i]
            return x

        def C(y, j):
            x = 0
            j = int(j)
            for i in range(j):
                x = x + i * i * y[i]
            return x

        threshold = 125  # Initial estimate
        Tprev = -2

        while threshold != Tprev:
            # Calculate some statistics
            mu = B(data, threshold) / A(data, threshold)
            nu = (B(data, 255)-B(data, threshold)) / (A(data, 255)-A(data, threshold))
            p = A(data, threshold) / A(data, 255)
            q = (A(data, 255)-A(data, threshold)) / A(data, 255)
            sigma2 = C(data, threshold) / A(data, threshold) - (mu * mu)
            tau2 = (C(data, 255) - C(data, threshold)) / (A(data, 255) - A(data, threshold)) - (nu * nu)

            # The terms of the quadratic equation to be solved.
            w0 = 1.0 / sigma2 - 1.0 / tau2
            w1 = mu / sigma2 - nu / tau2

            w2 = (mu * mu) / sigma2 - (nu * nu) / tau2 + log10((sigma2 * (q * q)) / (tau2 * (p * p)))

            # If the next threshold would be imaginary, return with the current one.
            sqterm = (w1 * w1) - w0 * w2
            # if sqterm < 0:
            #     print("MinError(I): not converging.")
            #     return threshold

            # The updated threshold is the integer part of the solution of the quadratic equation.
            Tprev = threshold
            temp = (w1 + sqrt(sqterm)) / w0

            if np.isnan(temp):
                print("MinError(I): NaN, not converging.")
                threshold = Tprev

            else:
                threshold = int(floor(temp))

        return threshold

    # Create histogram from image
    hist = np.histogram(img, bins=np.arange(256))[0]

    # Calculate threshold
    threshold = get_min_error_threshold(hist)

    # Use found threshold and get the mask
    _, mask = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    return mask


self = PreProcessing(name_csv=name_csv, input_path=input_path, output_path=output_path, image_type=image_type)


# Create Dataframe with slices information
columns = ["original_image", "image_path", "slice_path", "x_coord", "y_coord", "x_size", "y_size", "category"]
df_list = pd.DataFrame(columns=columns)

# Create output folder if not existent
if not os.path.exists(self.output_path):
    os.makedirs(self.output_path)

# Delete all files in output folder
for content in os.listdir(Path(self.output_path)):
    file_path = os.path.join(self.output_path, content)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))


for category_path in self.input_path.iterdir():
    # If path does not lead to a folder skip it
    if not category_path.is_dir():
        continue

    category = category_path.stem
    # Create path for the output of current category (directory=category)
    output_sub_path = Path(self.output_path, category)

    # Create output path for category
    output_sub_path.mkdir(exist_ok=True)

    # For every image in current category folder crop it and store the ones containing tissue
    for image_path in category_path.iterdir():
        # Read image
        img = cv2.imread(str(image_path))

        # Turns img into a 2D array according to its origin
        img = self.channel_selection(img)

        # Remove Linear Bias
        img = linear_bias_removal(img)

        # Remove noise using a Gaussian filter
        gaussian = 9
        if gaussian is not None:
            img = cv2.GaussianBlur(img, (gaussian, gaussian), cv2.BORDER_DEFAULT)

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.axis("off")

        methods = ["otsu", "triangle", "huang", "mean", "percentil_65", "percentil_75", "percentil_85"]

        ax1 = fig.add_subplot(2, ceil(len(methods) / 2), 1)
        ax1.imshow(img, cmap='gray')
        ax1.axis("off")
        ax1.title.set_text(f'Original Image')

        count = 2
        for method in methods:

            # Get mask after applying specific threshold method
            mask = get_tissue_detector_mask(img, method)

            ax1 = fig.add_subplot(2, ceil(len(methods) / 2), count)
            ax1.imshow(mask, cmap='gray')
            ax1.axis("off")
            ax1.title.set_text(f'{method}')

            count = count + 1

        plt.savefig(Path(output_sub_path, image_path.stem))
        plt.close()
