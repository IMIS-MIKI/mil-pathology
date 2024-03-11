import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import mil_pytorch.mil as mil


# TODO: Check default values
class BagGenerator:
    """Description for BagGenerator class
    # name_csv: Name of the csv file where all the information regarding patients, path for the slices and classification is
    #   stored
    # bags_per_class: Total Number of images/patients/bags to be generated (Size of the Dataset to be used for each class)
    # slices_per_bag: Number of slices per bag
    # batch_size: Batch size
    """

    def __init__(self, device, transforms_train = None, transforms_test=None, transforms_predict=None, classes: list = [],
                 name_csv: str = 'image-data.csv', bags_per_class: int = 20, slices_per_bag: int = 10, batch_size: int = 1, verbose: bool = False):
        # TODO: add input data validators and throw exceptions
        # Validate classes inputs
        # TODO: add inputs for interpretation (so far hardcoded)
        self.name_csv = name_csv
        self.bags_per_class = bags_per_class
        self.slices_per_bag = slices_per_bag
        self.batch_size = batch_size
        self.transforms_train = transforms_train
        self.transforms_test = transforms_test
        self.transforms_predict = transforms_predict

        self.device = device
        self.classes = classes
        self.active_cv_fold = None
        self.split_ratio = 0 # 0 is all test, 1 is all training

        if verbose:
            print("BagGenerator initialized successfully")

    def run(self, active_cv_fold=0, interpret=False, interpret_image_filter: str = ''):
        """Create dataset from dataframe:
        active_cv_fold: needs to explicitly be defined as None in case of inference model
        """

        if not self.valid():
            return [None, None]

        # Set active fold if set
        self.active_cv_fold = active_cv_fold

        # Load dataframe with slice information
        df = pd.read_csv(self.name_csv, usecols=["category", "original_image", "slice_path", "cv_fold"])

        # Split train and test DFs
        # active_cv_fold=None for inference model: return only one set/training set is useless here
        if self.active_cv_fold is None:
            # does not return training set:
            return None, self.createBagDataset(df, batch_size=self.batch_size, transforms=self.transforms_test, test_set=True, interpret=interpret, interpret_image_filter=interpret_image_filter)

        else:
            df_test = df[df["cv_fold"] == self.active_cv_fold]
            df_train = df[df["cv_fold"] != self.active_cv_fold]

        # Ratio between train and test dataset (is only set other than 0 if not inference)
        # this splits weighted by number of slices (undesired):
        #self.split_ratio = df_train.shape[0] / (df_train.shape[0] + df_test.shape[0])
        # this splits by target fraction (10% for 10-fold-cv):
        max_fold = int(df["cv_fold"].max())
        self.split_ratio = 1/max_fold if max_fold > 1 else 0 # change split_ratio only if more than 2 folds, e.g. when cv is active

        if df_test.shape[0] == 0:
            return self.createBagDataset(df_train, batch_size=self.batch_size, transforms=self.transforms_train), None

        return self.createBagDataset(df_train, batch_size=self.batch_size, transforms=self.transforms_train), \
               self.createBagDataset(df_test, batch_size=self.batch_size, transforms=self.transforms_test, test_set=True)  # temporarily added batch size here

    def run_predict(self):
        """Create dataset from dataframe:
        active_cv_fold: needs to explicitly be defined as None in case of inference model
        """

        # Load dataframe with slice information
        df = pd.read_csv(self.name_csv, usecols=["category", "original_image", "slice_path"])

        data = []
        ids = []
        labels = []
        original_images = []

        # Get dataframe with all possible images
        df_images = df[["original_image"]].drop_duplicates()

        count = 0
        for _, image in df_images.iterrows():
            original_image = image["original_image"]
            df_slice_image = df[df["original_image"] == original_image]

            while df_slice_image.shape[0] > self.slices_per_bag:
                sample = df_slice_image.sample(self.slices_per_bag)["slice_path"].tolist()

                # Remove sample from df to avoid duplicates
                df_slice_image = df_slice_image[~df_slice_image["slice_path"].isin(sample)]

                # For each image, select a set of slices and add them to the final dataset
                data.extend(sample)
                ids.extend(self.slices_per_bag*[count])

                # Save label (assumes 1st category is 0, but needs to be 0 in the future)
                labels.append(self.slices_per_bag*[0])

                original_images.append(original_image)
                count = count + 1

        # Generate dataset from data, ids and respective labels
        # data is each entry a slice
        # ids is each entry the bag to which it is associated
        # labels is the label for each bag

        mildataset = mil.MilDataset(np.array(data),  # torch.stack(data).to(self.device),  # data
                                    torch.tensor(ids, dtype=torch.long).to(self.device),
                                    torch.tensor(labels, dtype=torch.long).to(self.device),
                                    transforms=self.transforms_predict)

        # no need to shuffle in the test set:
        return DataLoader(mildataset, batch_size=self.batch_size, collate_fn=mil.collate, shuffle=False), original_images

    def createBagDataset(self, df: pd.DataFrame, batch_size: int = 10, transforms=None, test_set: bool = False, interpret: bool = False, interpret_filename: str = 'interpret.csv', interpret_image_filter: str = ''):
        """Create Bag dataset for the provided dataframe
            interpret: special sampling/dataloader for interpretation, only in combination with active_cv_fold is None (inference mode) and test_set==True
        """

        data = []
        ids = []
        labels = []

        if test_set:
            n_images = round(self.bags_per_class * (1 - self.split_ratio))
        else:
            n_images = round(self.bags_per_class * self.split_ratio)

        # interpretation mode: interpretation is done for every original_image
        if interpret:
            assert self.active_cv_fold is None
            assert test_set

            #TODO: add more sophisticated sampling scheme here:
            #       * also sample random bag y and compose combinations
            #       * allow for non-random sampling
            slices_per_subbag_x = 5 # number of tiles in bags of group x
            #slices_per_subbag_y = 0 # number of tiles in bags of group y

            #TODO: single image because of memory constrains
            if interpret_image_filter != '':
                df = df[df["original_image"] == interpret_image_filter]
            df_samples = df[["category", "original_image"]].drop_duplicates().sample(n_images, replace=True).reset_index()

            #### ALTERNATIVE, DESIRED SOLUTION:
            ## Create a custom function to replicate rows
            #def replicate_rows(group, n):
            #    return group.iloc[[0] * n]
            ## Group by "category" and "original_image" columns, then apply the custom function
            #df_samples = df.groupby(['category', 'original_image'], group_keys=False).apply(replicate_rows, n=n_images)

            # iterate over all individual images in the set
            for id_, sample in df_samples.iterrows():
                # For the specific image selected, select a set of slices and add them to the final dataset
                data.extend((df[df["original_image"] == sample["original_image"]].sample(slices_per_subbag_x, replace=True))["slice_path"].tolist())
                ids.extend(slices_per_subbag_x*[id_])
                labels.append(self.classes.index(str(sample["category"])))

            labels_expanded = [item for item in labels for _ in range(slices_per_subbag_x)]

            # create dataframe with bags info for interpretation here
            df_data = {
                    'bag_x': data,
                    'id': ids,
                    'label': labels_expanded
                    }
            df = pd.DataFrame(df_data)
            # group by id: one line per id
            df = df.groupby('id', as_index=True).agg({'bag_x': list, 'label': 'first'})
            df.to_csv(interpret_filename)

        else:
            # Get dataframe with all possible images per category (avoid being imbalanced by the quantity of images per
            # category / slices per image)
            df_samples = df[["category", "original_image"]].drop_duplicates()

            # Get a dataframe with images to be used for each category in the final dataset (there can be repetitions)
            df_samples = df_samples.groupby(["category"]).sample(n_images, replace=True).reset_index()

            for id_, sample in df_samples.iterrows():

                # # For the specific image selected, select a set of slices and add them to the final dataset
                # for slice_path in df[df["original_image"] == sample["original_image"]].sample(self.slices_per_bag, replace=True)[
                #         "slice_path"]:
                #     # Load slice as image as float (0..1) instead of int (0..255)
                #     img = torchvision.io.read_image(slice_path).float()/255
                #
                #     # Stack bag and save it
                #     data.append(img)
                #     ids.append(id_)

                # For the specific image selected, select a set of slices and add them to the final dataset
                data.extend((df[df["original_image"] == sample["original_image"]].sample(self.slices_per_bag, replace=True))["slice_path"].tolist())
                ids.extend(self.slices_per_bag*[id_])

                # Save label (assumes 1st category is 0, but needs to be 0 in the future)
                labels.append(self.classes.index(str(sample["category"])))

        # Generate dataset from data, ids and respective labels
        # data is each entry a slice
        # ids is each entry the bag to which it is associated
        # labels is the label for each bag

        mildataset = mil.MilDataset(np.array(data),  # torch.stack(data).to(self.device),  # data
                                    torch.tensor(ids, dtype=torch.long).to(self.device),
                                    torch.tensor(labels, dtype=torch.long).to(self.device),
                                    transforms=transforms)

        # no need to shuffle in the test set:
        return DataLoader(mildataset, batch_size=batch_size, collate_fn=mil.collate, shuffle=~test_set)

    def get_attributes(self):
        # Attribute list from BagGenerator
        return {'bags per class': self.bags_per_class, 'slices per bag': self.slices_per_bag, 'batch size': self.batch_size, 'classes': self.classes, 'active cv fold': self.active_cv_fold}

    def load_attributes(self, dict_attributes: list):
        self.bags_per_class = dict_attributes['bags per class']
        self.slices_per_bag = dict_attributes['slices per bag']
        self.batch_size = dict_attributes['batch size']
        self.classes = dict_attributes['classes']
        self.active_cv_fold = dict_attributes['active cv fold']

        self.classes = [elem[1:-1] for elem in str(self.classes[1:-1]).split(", ")]

        self.valid()

    def load_transforms(self, transforms_train, transforms_test):
        self.transforms_train = transforms_train
        self.transforms_test = transforms_test

    def valid(self, from_load=False):
        # TODO: Check how to validate device
        valid = True
        problems = []

        if self.name_csv is None or not isinstance(self.name_csv, str):
            valid = False
            problems += ["Name csv either is None or is not a string"]

        if not isinstance(self.classes, list) or any([not isinstance(elem, str) for elem in self.classes]) or \
                self.classes.__len__() < 2:
            valid = False
            problems += ["Class list with problems"]

        if not isinstance(self.bags_per_class, int) or self.bags_per_class < 2:
            valid = False
            problems += ["Bags per class with problems"]

        if not isinstance(self.slices_per_bag, int) or self.slices_per_bag < 1:
            valid = False
            problems += ["Slices per bag with problems"]

        if not isinstance(self.batch_size, int) or self.batch_size < 2:
            valid = False
            problems += ["Batch size with problems"]

        if not valid:
            print("BAG GENERATOR - VALIDATION PROBLEM:")
            for problem in problems:
                print("\n" + problem)

        return valid
