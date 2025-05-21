import subprocess
import os
import shutil
import cv2

tif_folder = "/home/mario/Projects/ki-project1/data/Camelyon/tif/training"
png_folder = "/home/mario/Projects/ki-project1/data/Camelyon/png/training"

resolution = 10  # [40, 10, 2.5, 0.625] <- Resolutions available -
# If the image is too big instead it is sampled with [20, 5, 1.25]


if not os.path.exists(png_folder):
    os.makedirs(png_folder)
# Delete all files in png folders
for filename in os.listdir(png_folder):
    file_path = os.path.join(png_folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))


# Transform the tiff file and store it in the destination folder
folder_files = os.listdir(tif_folder)
# Iterate over the images
for file_name in folder_files:
    if file_name == '.listing':
        continue
    print(file_name)
    # Remove ".ndpi" from string
    file_name = file_name[:-4]

    # Convert tiff to png
    img = cv2.imread(f'{tif_folder}/{file_name}.tif')

    # Convert from BGR (default from cv2) to HSV and drop all channels instead of (V)alue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]

    # Save new image in png folder
    cv2.imwrite(f'{png_folder}/{file_name}.png', img)
