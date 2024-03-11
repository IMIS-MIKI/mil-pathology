import random
import os
import shutil
import pandas as pd
import numpy as np
import cv2
import torchvision
import torch
from pathlib import Path, WindowsPath, PosixPath
from math import fabs, log


class PreProcessing:
    """ Description for pre-processing class

    # input_path: path containing category folders
    # output_path: Folder path to store the output, after slicing the images
    # tile_size: x, y dimensions of each slice
    # tile_shift: Shift applied to the slices when extracting crop (for data augmentation) - Instead of starting on pixel 0, starts
    #   on the one identified by the shift
    # tile_step: Step applied to the slices when extracting crop (for data augmentation) - Controls the overlapping between
    #   consecutive slices (positive number leads to overlap)
    # name_csv: Name of the csv file to be produced (where all the information regarding patients, path for the slices and
    #   classification is stored)
    # valid_background_perc: Required percentage of tissue, in a image, to be classified as valid for the classification process
    # cv_folds: Number of crossvalidation folds
    # split_ratio: Ratio to split training and test set (only if cv_folds==None)
    # magnitude = "_x10"
    """

    def __init__(self, input_path=Path("input"), output_path=Path("output"), tile_size=[224, 224], tile_shift=[0, 0], tile_step=[0, 0],
                 name_csv='image-data', valid_background_perc=0.9, cv_folds=None, split_ratio=0.7, verbose=False, image_type=None, magnitude="_x10"):
        # TODO: add input data validators and throw exceptions
        # Validate classes inputs
        self.input_path = input_path
        self.output_path = output_path
        self.tile_size = tile_size
        self.tile_shift = tile_shift
        self.tile_step = tile_step
        self.name_csv = name_csv
        self.valid_background_perc = valid_background_perc
        self.cv_folds = cv_folds
        self.split_ratio = split_ratio
        self.image_type = image_type
        self.magnitude = magnitude

        self.nb_input_channels = 0
        self.train_set_statistics = (None, None)
        self.verbose = verbose

        if os.name == "posix":
            self.dir_breaker = "/"
        elif os.name == "nt":
            self.dir_breaker = "\\"
        else:
            raise Exception("Only run on Windows or Linux")

        if self.verbose:
            print("Pre-processing initialized successfully")

    def run(self):
        """Analyse folders in input_path containing the different images (each folder contains a different
        category/class). These images are then cropped into slices and the ones containing tissue are stored.
        """

        if not self.valid():
            return

        # If file already exists no need to recreate it
        if os.path.exists(self.name_csv):
            return

        # Create Dataframe with slices information
        columns = ["original_image", "image_path", "slice_path", "x_coord", "y_coord", "x_size", "y_size", "category"]
        df_list = pd.DataFrame(columns=columns)

        slices_per_image = pd.DataFrame(columns=["category", "original_image", "nb_slices"])

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
                df_cropped_image = self.crop_image(image_path, output_sub_path, category, columns)

                df_list = pd.concat([df_list, df_cropped_image], ignore_index=True)

                slices_per_image = \
                    pd.concat([slices_per_image,
                               pd.DataFrame([[category,
                                              str(image_path).split(self.dir_breaker)[-1].split(self.magnitude)[0],
                                              df_cropped_image.shape[0]]],
                                            columns=["category", "original_image", "nb_slices"])])

                if self.verbose:
                    if df_cropped_image.shape[0] > 1:
                        print(f"{image_path} with {df_cropped_image.shape[0]} valid slices")
                    else:
                        print(f"{image_path} NO SLICES")

        slices_per_image = slices_per_image.groupby(["category", "original_image"])['nb_slices'].agg('sum').reset_index()
        # Assign test/train label to dataset
        df_list = self.split_dataset(df_list)

        # Save dataframe with all slices information
        df_list.to_csv(self.name_csv)
        slices_per_image.to_csv(self.name_csv + "_sli_per_img")

        if self.verbose:
            print("Pre-Processing run successfully and created the file: " + self.name_csv)

    def crop_image(self, input_image_path, output_dir, category, columns, kernel_dim_mask_cleaner=11, gaussian=9):
        """
        Crop image and compare with mask provided by OTSU method, in order to filter out slices without tissue
        """
        # Create Dataframe with slices information for current image
        df_slices = pd.DataFrame(columns=columns)

        # Read image
        img = cv2.imread(str(input_image_path))

        # Turns img into a 2D array according to its origin
        img = self.channel_selection(img)

        # TODO: check if it is called bias
        # Remove Linear Bias
        img = self.linear_bias_removal(img)

        # Remove noise using a Gaussian filter
        if gaussian is not None:
            img = cv2.GaussianBlur(img, (gaussian, gaussian), cv2.BORDER_DEFAULT)

        # Get mask
        mask = self.get_tissue_detector_mask(img, method="triangle")

        # Remove artifacts and fills holes
        mask = self.mask_cleaner(mask, kernel_dim_mask_cleaner)

        # TODO: Check last slice (not enough pixels)
        # Loop through all possible slice positions and store the valid ones (Using shift and step)
        for pos_x in range(self.tile_shift[0], img.shape[0] - self.tile_size[0],
                           self.tile_size[0] - self.tile_step[0]):
            for pos_y in range(self.tile_shift[1], img.shape[1] - self.tile_size[1],
                               self.tile_size[1] - self.tile_step[1]):

                # Crop mask and check if it has enough tissue in order to be valid, if not skip it
                mask_slice = mask[pos_x: pos_x + self.tile_size[1], pos_y: pos_y + self.tile_size[0]]

                if not self.has_tissue(mask_slice):
                    continue

                # Crop image to get slice
                slice_ = img[pos_x: pos_x + self.tile_size[1], pos_y: pos_y + self.tile_size[0]]

                # Save slice in output path
                output_slice_path = Path(output_dir, "{}_{}_{}.png".format(input_image_path.stem, pos_x, pos_y)) \
                    .with_suffix(".png")

                cv2.imwrite(str(output_slice_path), slice_)

                # Add slice information into dataframe
                df_slices = df_slices.append(pd.DataFrame([(str(input_image_path).split(self.dir_breaker)[-1].split(self.magnitude)[0],
                                                            str(input_image_path), str(output_slice_path), pos_x, pos_y,
                                                            self.tile_size[1], self.tile_size[0], category)],
                                                          columns=columns))

        return df_slices

    def get_train_set_statistics(self):
        """Return mean + std over all images in training set (read from csv)
        """

        if self.train_set_statistics[0] is not None:
            return self.train_set_statistics

        df = pd.read_csv(self.name_csv, usecols=["slice_path", "cv_fold"])
        df_train = df[df["cv_fold"] == 1]

        # https://www.thoughtco.com/sum-of-squares-formula-shortcut-3126266

        pixel_sum, pixel_squared_sum, n_pixels = 0, 0, 0
        for _, s in df_train.iterrows():
            img = torchvision.io.read_image(s["slice_path"]).float() / 255
            n_pixels += torch.numel(img)
            pixel_sum += torch.sum(img, dim=[1, 2])
            pixel_squared_sum += torch.sum(img ** 2, dim=[1, 2])

        # mean = Σ(xi)/n
        mean = pixel_sum / n_pixels
        # variance = Σ(xi^2)-(Σxi)^2/n
        std = (pixel_squared_sum - pixel_sum ** 2 / n_pixels) ** 0.5

        self.train_set_statistics = (mean, std)

        return self.train_set_statistics

    def get_tissue_detector_mask(self, img, method="triangle"):
        """ Apply defined tissue detector technique.
        More examples tested in the test/test_tissue_detectors.py
        img: image
        method: name of method to use
        """

        if method == "otsu":
            mask = self.get_otsu_tissue_detector_mask(img)

        elif method == "triangle":
            mask = self.get_triangle_tissue_detector_mask(img)

        elif method == "huang":
            mask = self.get_huang_tissue_detector_mask(img)

        return mask

    @staticmethod
    def get_otsu_tissue_detector_mask(img):
        """ Apply the Otsu technique, in order to differentiate tissue from background.
        Returns a mask where True/1 means background and False/0 means tissue.
        img: image
        """
        # "Inspired" from here: https://docs.opencv.org/4.5.1/d7/d4d/tutorial_py_thresholding.html
        # http://web-ext.u-aizu.ac.jp/course/bmclass/documents/otsu1979.pdf

        # Otsu thresholding and mask generation
        _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return mask

    @staticmethod
    def get_triangle_tissue_detector_mask(img):
        """ Apply the triangle threshold method.
        https://pythonmana.com/2021/10/20211009194205124c.html
        img: image
        """

        # Triangle thresholding and mask generation
        _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

        return mask

    @staticmethod
    def get_huang_tissue_detector_mask(img):
        """ Apply the triangle threshold method.
        img: image
        """

        # Adapted from https://github.com/dnhkng/Huang-Thresholding/blob/master/Threshold.py
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

            return threshold

        # Create histogram from image
        hist = np.histogram(img, bins=np.arange(256))[0]

        # Calculate threshold
        threshold = get_huang_threshold(hist)

        # Use found threshold and get the mask
        _, mask = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

        return mask

    @staticmethod
    def linear_bias_removal(img):
        """Estimates vertical and horizontal linear pattern and removes it from the image
        Currently only removing slope (Example C from remove_bias_preprocessing)
        img: image"""

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

    @staticmethod
    def mask_cleaner(mask, kernel_dim=11):
        """Fills holes and removes artifacts"""

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

        # from matplotlib import pyplot as plt
        # plt.close()
        # plt.subplot(1, 2, 1)
        # plt.imshow(mask, 'gray')
        # plt.subplot(1, 2, 2)
        # plt.imshow(mask_cleaned, 'gray')
        # plt.show()

        return mask_cleaned

    def has_tissue(self, mask):
        """Returns whether the mask has a bigger quantity of background than the one accepted"""

        # Normalize mask
        mask[mask > 0] = 1

        # Check percentage of background
        if self.image_type == "normal":
            # Background is 0, tissue is 1 (Background is white/light color)
            perc_background = np.mean(mask)
        elif self.image_type == "fluorescent":
            # Background is 1, tissue is 0 (Background is black/dark color)
            perc_background = 1 - np.mean(mask)

        # Validate percentage of background
        # if percentage of image with background is lower than valid_percentage_background return true
        return perc_background < self.valid_background_perc

    def split_dataset(self, dataset):
        """Split dataset into train and test dataset:
        if cv_folds != None: assign classes 0 - cv_folds-1
        else: assign classes 0, 1 according to split ratio
        """
        random.seed(10) # reproducibility

        if self.cv_folds != None:
            splitclasses = list(range(self.cv_folds))
            weights = [1]*len(splitclasses)
        else:
            # 0 = test, 1 = train
            splitclasses = [1, 0]

            # weights for randomized
            weights = (self.split_ratio, 1 - self.split_ratio)

        # Get Dataframe with images and category
        df = dataset[["original_image", "category"]].drop_duplicates()

        # insert cv_fold column
        df['cv_fold'] = 0

        # assign train/test value to every element of subset of DF, where each subset correspondents to a single patient
        for category in df[["category"]].drop_duplicates().to_numpy().flatten():
            df.loc[df["category"] == category, ['cv_fold']] = random.choices(splitclasses, weights=weights,
                                                                             k=df[df["category"] == category].shape[0])

        return pd.merge(dataset, df, how='inner', on=['original_image', 'category'])

    def get_classes_name(self):
        list_cat = []
        for category_path in self.input_path.iterdir():
            # If path does not lead to a folder skip it
            if not category_path.is_dir():
                continue
            list_cat.append(str(category_path.stem))
        return list_cat

    def get_nb_input_channels(self):
        """Get number of channels in slices of current dataset"""

        if not self.nb_input_channels:
            df = pd.read_csv(self.name_csv, usecols=["slice_path"], nrows=1)

            # Get sample image to determine the number of channels in a slice
            self.nb_input_channels = torchvision.io.read_image(df.slice_path[0]).shape[0]

        return self.nb_input_channels

    def get_attributes(self):
        # Attribute dict from PreProcessing
        return {'input path': str(self.input_path), 'output path': str(self.output_path), 'tile size': self.tile_size, 'tile shift': self.tile_shift, 'tile step': self.tile_step,
                'name csv': self.name_csv, 'valid background perc': self.valid_background_perc, 'cv folds': self.cv_folds, 'split ratio': self.split_ratio, 'nb input channels': self.nb_input_channels, 'image type': self.image_type}

    def load_attributes(self, dict_attributes):
        self.input_path = Path(dict_attributes['input path'])
        self.output_path = Path(dict_attributes['output path'])
        self.tile_size = [int(elem) for elem in str(dict_attributes['tile size'][1:-1]).split(", ")]
        self.tile_shift = [int(elem) for elem in str(dict_attributes['tile shift'][1:-1]).split(", ")]
        self.tile_step = [int(elem) for elem in str(dict_attributes['tile step'][1:-1]).split(", ")]
        self.name_csv = dict_attributes['name csv']
        self.valid_background_perc = dict_attributes['valid background perc']
        self.cv_folds = dict_attributes['cv folds']
        self.split_ratio = dict_attributes['split ratio']
        self.nb_input_channels = dict_attributes['nb input channels']
        # catching exception for compatibility reasons:
        try:
            self.image_type = dict_attributes['image type']
        except(KeyError):
            self.image_type = None

        self.valid(from_load=True)

    def valid(self, from_load=False):
        valid = True
        problems = []

        if not from_load and (self.input_path is None or not isinstance(self.input_path, (WindowsPath, PosixPath))):
            valid = False
            problems += ["Input path either is None or is not a string"]

        if not from_load and (self.output_path is None or not isinstance(self.output_path, (WindowsPath, PosixPath))):
            valid = False
            problems += ["Output path either is None or is not a string"]

        # TODO: check if there is a max value for tile size | self.tile_size[X] > 512
        if not isinstance(self.tile_size, list) or \
                not isinstance(self.tile_size[0], int) or self.tile_size[0] < 0 or \
                not isinstance(self.tile_size[1], int) or self.tile_size[1] < 0:
            valid = False
            problems += ["Tile size with problems"]

        # TODO: check if there is a max value for tile shift | self.tile_shift[X] > 512
        if not isinstance(self.tile_shift, list) or \
                not isinstance(self.tile_shift[0], int) or self.tile_shift[0] < 0 or \
                not isinstance(self.tile_shift[1], int) or self.tile_shift[0] < 0:
            valid = False
            problems += ["Tile shift with problems"]

        # TODO: check if there is a max/min value for tile step | self.tile_sstep[X] > 512
        if not isinstance(self.tile_step, list) or \
                not isinstance(self.tile_step[0], int) or \
                not isinstance(self.tile_step[1], int):
            valid = False
            problems += ["Tile step with problems"]

        if self.name_csv is None or not isinstance(self.name_csv, str):
            valid = False
            problems += ["Name csv for preprocessing output either is None or is not a string"]

        if self.valid_background_perc is None or not isinstance(self.valid_background_perc, float) or \
                self.valid_background_perc < 0 or self.valid_background_perc >= 1:
            valid = False
            problems += ["Valid background has problems (should be value between 0 and 1)"]

        if self.split_ratio is None or not isinstance(self.split_ratio, float) or \
                self.split_ratio < 0 or self.split_ratio > 1:
            valid = False
            problems += ["Split ratio has problems (should be value between 0 and 1)"]

        if not valid:
            print("PREPROCESSING - VALIDATION PROBLEM:")
            for problem in problems:
                print("\n" + problem)

        return valid
