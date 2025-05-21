# Analyse mask cleaner dimensions
# This will expand the tissue and contract it and then contract tissue and expand it again in the mask
# This allows to fill holes in the first phase and in the second round it deletes artifacts from the mask
# Analysed using images with x10 resolution

# Results
# Selected 11

import os
import shutil
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from matplotlib import pyplot as plt

# preprocessing parameters
input_path = Path("data/png_files")
output_path = Path("data/test/check_mask_cleaner_dim")

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
        # elif self.image_type == "fluorescent":
        #     color_convert = None
        #     channel = 2

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


def get_otsu_tissue_detector_mask(img, gaussian=9, visualize=False):
    """ Apply the Otsu technique, in order to differentiate tissue from background.
    Returns a mask where True/1 means background and False/0 means tissue.
    slice_path: string with path to the image
    gaussian: blur parameter in pixels, None for no blur
    invert_list: list of channels (same shape as channel_list) to invert, None for no inversion
    """
    # "Inspired" from here: https://docs.opencv.org/4.5.1/d7/d4d/tutorial_py_thresholding.html
    # http://web-ext.u-aizu.ac.jp/course/bmclass/documents/otsu1979.pdf

    # Remove noise using a Gaussian filter
    if gaussian is not None:
        img = cv2.GaussianBlur(img, (gaussian, gaussian), cv2.BORDER_DEFAULT)

    # Otsu thresholding and mask generation
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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

        # Get mask after applying OTSU method
        mask = get_otsu_tissue_detector_mask(img)

        # Remove artifacts and fills holes

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.axis("off")

        ax1 = fig.add_subplot(2, 3, 1)
        ax1.imshow(img, cmap='gray')
        ax1.axis("off")
        ax1.title.set_text(f'Original Image')

        count = 2
        for kernel_dim in [5, 7, 9, 11, 13]:
            kernel = np.ones((kernel_dim, kernel_dim))

            # Fill Holes
            # Expand Tissue
            mask_cleaned = cv2.filter2D(mask, ddepth=-1, kernel=kernel)
            # Invert
            mask_cleaned = -(mask_cleaned - 255)
            # Concentrate
            mask_cleaned = cv2.filter2D(mask_cleaned, ddepth=-1, kernel=kernel)

            # Remove artifacts (Similar to above, but we expand the background)
            # Expand Background
            mask_cleaned = cv2.filter2D(mask_cleaned, ddepth=-1, kernel=kernel)
            # Invert (Since it is applied to the original inverted, it goes back to normal)
            mask_cleaned = -(mask_cleaned - 255)
            # Concentrate
            mask_cleaned = cv2.filter2D(mask_cleaned, ddepth=-1, kernel=kernel)

            ax1 = fig.add_subplot(2, 3, count)
            ax1.imshow(mask_cleaned, cmap='gray')
            ax1.axis("off")
            ax1.title.set_text(f'Kernel size = {kernel_dim}')

            count = count + 1

        plt.savefig(Path(output_sub_path, image_path.stem))
        plt.close()
