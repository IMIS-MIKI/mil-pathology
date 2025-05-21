# Analyse remove bias preprocessing
# Check which techniques are better to remove the influence from inconsistencies in the images. More precisely when there is a smooth gradient.

# Results
# We will be testing using C (remove slope of linear regression) and then D (C followed by removing the median) so that we can check the results and choose the optimal one
# F (D and then renormalized) is a bit useless since we later renormalize everything

import os
import shutil
import pandas as pd
import numpy as np
import cv2
import torchvision
import torch
from pathlib import Path, WindowsPath, PosixPath
from pathlib import Path
import time

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# preprocessing parameters

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
        # TODO: add input data validators and throw exceptions
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


input_path = Path("data/png_files")
output_path = Path("data/test/remove_bias_preprocessing")
valid_background_perc = 0.4
image_type = "fluorescent"

self = PreProcessing(input_path=input_path, output_path=output_path, valid_background_perc=valid_background_perc,
                     image_type=image_type)

for content in os.listdir(Path(self.output_path)):
    file_path = os.path.join(self.output_path, content)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

# Run

for category_path in self.input_path.iterdir():
    # if category_path.stem != "3":
    #     continue

    category = category_path.stem
    # Create path for the output of current category (directory=category)
    output_sub_path = Path(self.output_path, category)

    # Create output path for category
    output_sub_path.mkdir(exist_ok=True)

    # For every image in current category folder crop it and store the ones containing tissue
    for image_path in category_path.iterdir():
        # if image_path.stem != "AHC179_Kongo_FL_x10_z0":
        #     continue

        # Read image
        img = cv2.imread(str(image_path))

        # Turns img into a 2D array according to its origin
        img = img[:, :, 2]

        orig_img = img
        img_minus_regression = img.copy()
        img_minus_slope_regression = img.copy()

        # Remove Linear Bias

        # Remove linear Horizontal influence
        model = np.polyfit(range(img.shape[1]), np.median(img, 0), 1)

        for i in range(img.shape[1]):
            # Minus regression
            new_array_reg = img_minus_regression[:, i] - (i * model[0] + model[1])
            new_array_reg[new_array_reg < 0] = 0
            img_minus_regression[:, i] = new_array_reg

            # Minus slope regression
            new_array_slope = img_minus_slope_regression[:, i] - (i * model[0])
            new_array_slope[new_array_slope < 0] = 0
            img_minus_slope_regression[:, i] = new_array_slope

        # Remove linear vertical influence
        model = np.polyfit(range(img.shape[0]), np.median(img, 1), 1)

        for i in range(img.shape[0]):
            # Minus regression
            new_array_reg = img_minus_regression[i, :] - (i * model[0] + model[1])
            new_array_reg[new_array_reg < 0] = 0
            img_minus_regression[i, :] = new_array_reg

            # Minus slope regression
            new_array_slope = img_minus_slope_regression[i, :] - (i * model[0])
            new_array_slope[new_array_slope < 0] = 0
            img_minus_slope_regression[i, :] = new_array_slope

        # After removing the slope also remove the median
        img_minus_slope_regression_minus_full_background = img_minus_slope_regression.copy() - np.median(img_minus_slope_regression)
        img_minus_slope_regression_minus_full_background[img_minus_slope_regression_minus_full_background < 0] = 0

        # Readd median to not background (0)
        img_readd_background = img_minus_slope_regression_minus_full_background.copy()
        img_readd_background[img_readd_background > 0] = img_readd_background[img_readd_background > 0] + np.median(img_minus_slope_regression)

        # Renormalize (new_value = current_value * old_max / current_max )
        img_renormalize = img_minus_slope_regression_minus_full_background.copy()
        img_renormalize = img_renormalize * (np.max(img_minus_slope_regression) / np.max(img_minus_slope_regression_minus_full_background))


        fig, ax = plt.subplots(figsize=(10, 10))
        ax.axis("off")
        ax1 = fig.add_subplot(3, 3, 2)
        ax1.imshow(img, cmap='gray')
        ax1.axis("off")
        ax1.title.set_text('(A) Original')
        ax2 = fig.add_subplot(3, 3, 4)
        ax2.imshow(img_minus_regression, cmap='gray')
        ax2.axis("off")
        ax2.title.set_text('(B) Removing Regression')
        ax3 = fig.add_subplot(3, 3, 5)
        ax3.imshow(img_minus_slope_regression, cmap='gray')
        ax3.axis("off")
        ax3.title.set_text('(C) Removing Slope Reg')
        ax4 = fig.add_subplot(3, 3, 6)
        ax4.imshow(img_minus_slope_regression_minus_full_background, cmap='gray')
        ax4.axis("off")
        ax4.title.set_text('(D) Removing median from C')
        ax5 = fig.add_subplot(3, 3, 7)
        ax5.imshow(img_readd_background, cmap='gray')
        ax5.axis("off")
        ax5.title.set_text('(E) Readd median to D (not Background)')
        ax6 = fig.add_subplot(3, 3, 9)
        ax6.imshow(img_renormalize, cmap='gray')
        ax6.axis("off")
        ax6.title.set_text('(F) Renormalize D')

        # plt.show()

        plt.savefig(Path(output_sub_path, image_path.stem))
        plt.close()

        time.sleep(1)
