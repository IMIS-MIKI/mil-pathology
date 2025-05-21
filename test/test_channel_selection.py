# Analyse import and respective channels and check which one we should use

# Results
# We should convert images from BGR to HSV and then select the Value channel
# Hue and Saturation seem to be partially random with no important information


import os
import shutil
import cv2
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm

input_path = Path("data/png_files")
output_path = Path("data/test/test_channel_selection")
valid_background_perc = 0.4
image_type = "fluorescent"

for content in os.listdir(Path(output_path)):
    file_path = os.path.join(output_path, content)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

# Run

tiff_folder = "data/tiff_files"
for category_path in Path(tiff_folder).iterdir():

    category = category_path.stem
    # Create path for the output of current category (directory=category)
    output_sub_path = Path(output_path, category)

    # Create output path for category
    output_sub_path.mkdir(exist_ok=True)

    for full_file_name in Path(category_path).iterdir():
        # TODO: Not sure if this is really needed or if we should just use the tif instead of the png
        # Convert tiff to png
        img_orig = cv2.imread(str(full_file_name))
        cv2.imwrite(str(Path(output_path, category_path.stem, full_file_name.stem + "_bgr.png")), img_orig)  # Channel order is BGR [0, 0, 38] - initial pixels

        for channel in [0, 1, 2]:
            cv2.imwrite(str(Path(output_path, category_path.stem, full_file_name.stem + f"_bgr_{channel}.png")),
                        img_orig[:, :, channel])  # Channel order is BGR [0, 0, 38]

        # Convert from BGR to HSV and drop all channels instead of (V)alue
        img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2HSV)

        cv2.imwrite(str(Path(output_path, category_path.stem, full_file_name.stem + f"_hsv.png")),
                    img)

        for channel in [0, 1, 2]:
            cv2.imwrite(str(Path(output_path, category_path.stem, full_file_name.stem + f"_hsv_{channel}.png")),
                        img[:, :, channel])
