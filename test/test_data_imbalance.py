from utils.preprocessing import PreProcessing
from pathlib import Path
from utils.visualize import show_dataset_statistics

# preprocessing parameters

# input_path = Path("kongo-ndpi_png_x10")
# output_path = Path("data/output/kongo-ndpi_png_x10")

input_path = Path("data/all/png")
output_path = Path("data/all/output")
name_csv = 'data/all/data_preprocessing_all_x10_1vs23'

# name_csv = 'data_preprocessing_kongo_x10'

perform_preprocessing = False
train_test_split = 0.8
valid_background_perc = 0.4

# define image dye type. normal="normal", kongo-red="fluorescent"
image_type = "fluorescent"

##########################

# Pre-Processing. Train/Test splitting happens here. maybe reconsider
preprocessing = PreProcessing(name_csv=name_csv, input_path=input_path, output_path=output_path,
                              split_ratio=train_test_split, valid_background_perc=valid_background_perc, verbose=True,
                              image_type=image_type)
if perform_preprocessing:
    preprocessing.run()
show_dataset_statistics(name_csv + "_sli_per_img")
