from torch import Tensor
import torch.nn as nn
import torchvision.transforms as T
import torch
import numbers
import random


class BagTransform(object):
    """Apply given transformation to slices in bag individually.
    """

    def __init__(self, input_transform=None, all_orientations=False, color_jitter=False, normalize=[None, None],
                 nb_input_channels=0):
        """Initializer for empty BagTransform
        """
        self.transform = input_transform
        self.all_orientations = all_orientations
        self.color_jitter = color_jitter
        self.normalize = normalize
        self.nb_input_channels = nb_input_channels

    def autobuild(self):

        if not self.valid():
            return

        transforms = []
        if self.all_orientations:
            transforms += [T.RandomHorizontalFlip(p=0.5),
                            T.RandomVerticalFlip(p=0.5),
                            T.RandomApply([T.RandomRotation((90, 90))], p=0.5)]

        if self.color_jitter and self.nb_input_channels:
            if self.nb_input_channels == 3:
                transforms += [T.ColorJitter(brightness=.5, contrast=0.2, saturation=.1, hue=.05)]
            else:
                transforms += [ColorJitter_1D(jitter_intensity=0.05, pixel_wise=False)]

        if self.normalize[0] is not None:
            transforms += [T.Normalize(self.normalize[0], self.normalize[1])]

        self.transform = nn.Sequential(*transforms)

    def __call__(self, bag_tensor):
        for i in range(len(bag_tensor)):
            bag_tensor[i, :, :, :] = self.transform(bag_tensor[i, :, :, :])
        return bag_tensor

    def get_attributes(self):
        # TODO: check how to encode the input_transform
        # Attribute dict from BagTransform
        if self.normalize[0] is None:
            return {'all orientations': self.all_orientations, 'color jitter': self.color_jitter, 'normalize mean': None, 'normalize std': None}

        return {'all orientations': self.all_orientations, 'color jitter': self.color_jitter, 'normalize mean': self.normalize[0].tolist(), 'normalize std': self.normalize[1].tolist()}

    def load_attributes(self, dict_attributes, is_test=True):

        self.all_orientations = dict_attributes['all orientations']
        self.color_jitter = dict_attributes['color jitter']
        self.normalize[0] = dict_attributes['normalize mean']
        self.normalize[1] = dict_attributes['normalize std']

        if is_test:
            self.all_orientations = False
            self.color_jitter = False
        else:
            self.all_orientations = bool(self.all_orientations)
            self.color_jitter = bool(self.color_jitter)

        # TODO: check how none is saved and if this is correct
        if self.normalize[0] == "None":
            self.normalize[0] = None
            self.normalize[1] = None
        else:
            self.normalize[0] = Tensor([float(elem) for elem in str(self.normalize[0][1:-1]).split(", ")])
            self.normalize[1] = Tensor([float(elem) for elem in str(self.normalize[1][1:-1]).split(", ")])

        self.valid(from_load=True)

    def valid(self, from_load=False):

        valid = True
        problems = []

        # TODO: Add validation for input_transform

        if self.all_orientations is None or not isinstance(self.all_orientations, bool):
            valid = False
            problems += ["All Orientations is not properly defined"]

        if self.color_jitter is None or not isinstance(self.color_jitter, bool):
            valid = False
            problems += ["Color jitter is not properly defined"]

        if not isinstance(self.normalize, (tuple, list)):
            valid = False
            problems += ["Normalize not a list"]
        else:
            mean = self.normalize[0]
            std = self.normalize[1]

            if not (mean is None or isinstance(mean, Tensor)) or \
                    any([not isinstance(value, Tensor) for value in mean]) or any([value < 0 for value in mean]):
                valid = False
                problems += ["Mean with problems"]

            if not (std is None or isinstance(std, Tensor)) or \
                    any([not isinstance(value, Tensor) for value in std]) or any([value < 0 for value in std]):
                valid = False
                problems += ["Std with problems"]

        if not valid:
            print("BAG TRANSFORM - VALIDATION PROBLEM:")
            for problem in problems:
                print("\n" + problem)

        return valid


class ColorJitter_1D(torch.nn.Module):
    """Randomly change the intensity of a grayscale image.
    If the image is torch Tensor, it is expected
    to have [..., 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, mode "1", "L", "I", "F" and modes with transparency (alpha channel) are not supported.

    Args:
        jitter_intensity (float or tuple of float (min, max)): Factor to change the intensity of pixels
    """

    def __init__(self, jitter_intensity=0, pixel_wise=False):
        super().__init__()
        self.jitter_intensity = self._check_input(jitter_intensity, 'jitter_intensity')
        self.pixel_wise = self._check_input(pixel_wise, 'pixel_wise')

    @torch.jit.unused
    def _check_input(self, value, name, bound=(0, float('inf'))):
        if name == "pixel_wise":
            if isinstance(value, bool):
                return value

        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            center = 1
            value = [center - float(value), center + float(value)]
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))

        if value[0] == value[1]:
            value = None

        return value

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        if self.jitter_intensity is None:
            return img

        if self.pixel_wise:
            jitter_tensor = torch.empty(img.shape).uniform_(self.jitter_intensity[0], self.jitter_intensity[1])
            return img * jitter_tensor
        else:
            return img * random.uniform(self.jitter_intensity[0], self.jitter_intensity[1])

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'jitter_intensity={0}'.format(self.jitter_intensity)
        return format_string + ')'
