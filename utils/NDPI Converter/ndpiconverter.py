import subprocess
import os
import shutil
import cv2

ndpi_folder = "C:/Users/Mario Macedo/Documents/PATHO_IMG/full_sets/ndpi"
tiff_folder = "C:/Users/Mario Macedo/Documents/PATHO_IMG/full_sets/tiff"
png_folder = "C:/Users/Mario Macedo/Documents/PATHO_IMG/full_sets/png"

resolution = 10  # [40, 10, 2.5, 0.625] <- Resolutions available -
# If the image is too big instead it is sampled with [20, 5, 1.25]


def runNDPISplit(resolution, ndpi_folder, tiff_folder, folder_class_name, file_name):
    subprocess.call(["utils/NDPI Converter/ndpisplit.exe",
                     "-v", "-TE",  # Verbose and show errors
                     "-m1024j100",
                     # m4096 - If file is bigger than 1Gb than it splits into a mosaic && j100 - keep 100% of quality
                     f"-x{resolution}",  # only store the resolution provided
                     "-O", f"{tiff_folder}/{folder_class_name}",  # Destination folder
                     f"{ndpi_folder}/{folder_class_name}/{file_name}.ndpi"])


def getFileNames(tiff_folder, folder_class_name, file_name, resolution):
    # Check if only one images was generated, or a mosaic (when file is too big cv2 cannot handle it, so we need to
    # split it into smaller parts)
    if os.path.exists(f'{tiff_folder}/{folder_class_name}/{file_name}_x{resolution}_z0_i1j1.tif'):
        # Delete full tif file
        os.unlink(f'{tiff_folder}/{folder_class_name}/{file_name}_x{resolution}_z0.tif')

        # Create list of partial files
        all_files = os.listdir(f'{tiff_folder}/{folder_class_name}')
        return [file for file in all_files if file.startswith(file_name)]
    elif os.path.exists(f'{tiff_folder}/{folder_class_name}/{file_name}_x{resolution}_z0.tif'):
        return [f'{file_name}_x{resolution}_z0.tif']


# Delete all files in tiff and png folders
for folder in [tiff_folder, png_folder]:
    if not os.path.exists(tiff_folder):
        os.makedirs(tiff_folder)
    if not os.path.exists(png_folder):
        os.makedirs(png_folder)

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

# Go to each category/cass folder and transform the ndpi file into a tiff file and store it in the destination folder
src_folders = os.listdir(ndpi_folder)
for folder_class_name in src_folders:
    src_folder_path = os.path.join(ndpi_folder, folder_class_name)

    # Generate class folders in tiff and png main folders
    tiff_folder_path = os.path.join(tiff_folder, folder_class_name)
    os.mkdir(tiff_folder_path)
    png_folder_path = os.path.join(png_folder, folder_class_name)
    os.mkdir(png_folder_path)

    folder_files = os.listdir(src_folder_path)
    for file_name in folder_files:

        if not file_name.endswith(".ndpi"):
            continue

        # Remove ".ndpi" from string
        file_name = file_name[:-5]

        # Generates tiff file from the ndpi file
        runNDPISplit(resolution, ndpi_folder, tiff_folder, folder_class_name, file_name)

        # Get file names
        list_image_generated = getFileNames(tiff_folder, folder_class_name, file_name, resolution)

        resolution_doubled = False

        if list_image_generated is None:
            print(f"{folder_class_name}/{file_name} current resolution not available, trying to generate from double")

            # Repeat two previous steps but with a different resolution (twice the original)
            runNDPISplit(2 * resolution, ndpi_folder, tiff_folder, folder_class_name, file_name)
            list_image_generated = getFileNames(tiff_folder, folder_class_name, file_name, 2 * resolution)
            resolution_doubled = True

            if list_image_generated is None:
                print(f"{folder_class_name}/{file_name} current resolution * 2 also not available, skipping image!!!!!")
                continue

        # Iterate over the images
        for full_file_name in list_image_generated:
            # Remove ".ndpi" from string
            full_file_name = full_file_name[:-4]
            # TODO: Not sure if this is really needed or if we should just use the tif instead of the png
            # Convert tiff to png
            img = cv2.imread(f'{tiff_folder}/{folder_class_name}/{full_file_name}.tif')

            # Convert from BGR (default from cv2) to HSV and drop all channels instead of (V)alue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]

            # Update resolution in case the "double" one had to be used
            if resolution_doubled:
                img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                full_file_name = full_file_name.replace(f"_x{2 * resolution}_", f"_x{resolution}_")

            # Save new image in png folder
            cv2.imwrite(f'{png_folder}/{folder_class_name}/{full_file_name}.png', img)

# Interesting options from ndpisplit
# -s  subdivide image into scanned zones (remove blank filling)
# -m4096J  Saves images up to 4096Mb as jpeg files - J did not seem to work, still making tif files and not jpeg
