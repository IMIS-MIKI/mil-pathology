import os
from utils.model import Model
from utils.preprocessing import PreProcessing
from utils.baggenerator import BagGenerator
from utils.bagtransform import BagTransform
from utils.modelinstance import ModelInstance
from utils.modelgovernance import ModelGovernance
from pathlib import Path
import pandas as pd
import gc
# import argparse
import torch
import itertools

# preprocessing parameters
input_path = Path("data/TC1/png")
output_path = Path("data/TC1/output")
name_csv = 'data/TC1/png/data_preprocessing_all_x10_1vs23'

# define image dye type. normal="normal", kongo-red="fluorescent"
image_type = "fluorescent"

normalize_globally = True  # with training set statistics

###### Parameters ######
slices_per_bag = 15
perc_background = 0.5
train_test_split = 0.9999
##########################

torch.cuda.empty_cache()
gc.collect()

# Load Model
model_governance = ModelGovernance()

# See current models
# previous_models = model_governance.get_models()

# Select model
model_instance = model_governance.get_instance(65)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Info
current_name_csv = name_csv + "_" + str(perc_background)
current_output_path = output_path / str(perc_background)

# Preprocessing
preprocessing = PreProcessing(name_csv=current_name_csv, input_path=input_path, output_path=current_output_path,
                                  split_ratio=train_test_split, valid_background_perc=perc_background, verbose=True,
                                  image_type=image_type)

if not os.path.exists(preprocessing.get_attributes()['output path']):
    preprocessing.run()


# Get number of channels from input images & name of the classes
nb_input_channels = preprocessing.get_nb_input_channels()

# Data Augmentation - Train vs Test & Train
transforms_predict = BagTransform()
transforms_predict.load_attributes(model_instance.get_info_by_group(prefix_select="TRANSFORM"), is_test=True)
transforms_predict.autobuild()


# Load Data from baggenerator.py
bag_generator = BagGenerator(device, transforms_predict=transforms_predict, name_csv=current_name_csv,
                             slices_per_bag=15, batch_size=1)

predict_dl, original_images = bag_generator.run_predict()

# Generate Model
model = Model(device, nb_input_channels=nb_input_channels, nb_classes=2)
model.load_attributes(model_instance.get_info_by_group())
model.initialize()

# Predict
predictions = model.predict(predict_dl)

# Cross information between prediction and original image
df_predict = pd.DataFrame(columns=["image", "prediction"])

df_predict["image"] = original_images
df_predict["prediction"] = predictions.tolist()

# why not working? :(
# a = df_predict.groupby(["image", "prediction"]).count()

res = []
for _, row in df_predict[["image"]].drop_duplicates().iterrows():
    image = row["image"]
    count_0 = df_predict[(df_predict["image"] == image) & (df_predict["prediction"] == 0)].shape[0]
    count_1 = df_predict[(df_predict["image"] == image) & (df_predict["prediction"] == 1)].shape[0]
    label = "1" if count_0 > count_1 else "23"
    res.append([image, count_0, count_1, label])

df_predict_final = pd.DataFrame(res, columns=["image", "count_0", "count_1", "type"])

df_predict_final.to_csv(Path("data/TC1/png/intermediate_65_result_TC1.csv"))
