import os
from utils.model import Model
from utils.preprocessing import PreProcessing
from utils.baggenerator import BagGenerator
from utils.bagtransform import BagTransform
from utils.modelinstance import ModelInstance
from utils.modelgovernance import ModelGovernance
from utils.visualize import show_interpretability_plot
from pathlib import Path
# import argparse
import torch

# for confidence plot
import numpy as np

# for get_train_set_statistics
import pandas as pd

model_governance = ModelGovernance()

# See current models
# previous_models = model_governance.get_models()

# Select model
model_instance = model_governance.get_instance(73)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Info
# Preprocessing
preprocessing = PreProcessing()
preprocessing.load_attributes(model_instance.get_info_by_group(prefix_select="PREPROCESSING"))
if not os.path.exists(preprocessing.get_attributes()['output path']):
    preprocessing.run()

# Get number of channels from input images & name of the classes
nb_input_channels = preprocessing.get_nb_input_channels()
classes = preprocessing.get_classes_name()
nb_classes = len(classes)

# Bag Transform
transforms_train = BagTransform(nb_input_channels=nb_input_channels)
transforms_train.load_attributes(model_instance.get_info_by_group(), is_test=False)
transforms_train.autobuild()

transforms_test = BagTransform()
transforms_test.load_attributes(model_instance.get_info_by_group(), is_test=True)
transforms_test.autobuild()

# Bag Generator
bag_generator = BagGenerator(device, transforms_train, transforms_test)
bag_generator.load_attributes(model_instance.get_info_by_group())
#train_dl, test_dl = bag_generator.run()

# Model
model = Model(device, nb_input_channels=nb_input_channels, nb_classes=nb_classes)
model.load_attributes(model_instance.get_info_by_group())
model.initialize()
#model.train(train_dl)

## Interpretability
# create dataloader
bag_generator.bags_per_class = 2000
_, test_dl = bag_generator.run(active_cv_fold=None, interpret=True, interpret_image_filter='KHAS20025_Kongo_H5_20x-Alexa 555')
# run interpretability inference
model.interpret_test(test_dl)
# run visualize
show_interpretability_plot(model_instance.get_info_by_group()["name csv"])

