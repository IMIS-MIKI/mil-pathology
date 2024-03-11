import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.utils as utils
import pandas as pd
import seaborn as sns
import ast
import re
import cv2

def show_bag(batch, train_stats):
    # input: batch from dataloader, train_stats (mean, std) from training set
    # output: plots of slices of first bag in batch, label. also predicted label maybe.

    # function to show a bag
    def imshow(img):
        npimg = img.numpy()
        plt.figure(figsize=(18, 100))
        plt.tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False)
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # get data from batch
    images, ids, labels = batch

    # split off first bag
    mask = (ids == ids.flatten()[0]).flatten()
    images = images[mask].cpu()
    labels = labels[0].cpu().numpy()

    # unnormalize images
    if train_stats is not None:
        images.mul_(train_stats[1].view(1, 3, 1, 1)).add_(train_stats[0].view(1, 3, 1, 1))

    # show images
    imshow(utils.make_grid(images, nrow=4))
    # print labels
    print('Bag label:', labels)

def show_dataset_statistics(name_csv):
    """Output statistics of the dataset read from csv named "name_csv"
    Each category with its own plot
    """
    ds_csv = pd.read_csv(name_csv)[["category", "original_image", "nb_slices"]]

    nb_cat = ds_csv["category"].drop_duplicates().size

    fig, axes = plt.subplots(1, nb_cat, figsize=(nb_cat*5, 5))
    sns.set_theme()
    pos = 1
    for grpname, grp in ds_csv.groupby("category"):

        axes[pos-1].hist(grp.groupby("original_image")["nb_slices"].agg("sum"), bins=[0+i for i in range(0, 150, 10)])

        axes[pos-1].set(xlabel='# Slices per image (S/I)',
                        ylabel='# Images',
                        title=f'Category {grpname}')

        text = grp.groupby(["original_image"])["nb_slices"].agg("sum").describe().\
            to_string(float_format=lambda x: f"{x:.0f}") + f'\nZeros     {grp[grp["nb_slices"] == 0].shape[0]}'

        axes[pos-1].text(1, 1, text, ha='right', va='top', transform=axes[pos-1].transAxes)
        pos = pos + 1

    if not os.path.isdir("data/test"):
        os.makedirs("data/test")
    plt.savefig(f"data/test/data-distribution.png")
    # plt.close()

def show_confidence_plot(predicted_class, target_class, confidence, bins=10):
    """ Group predicted confidence in bins groups. For every group, calculate actual accuracy. Create scatterplot of predicted accuracy = confidence and actual accurarcy.
    """
    accuracy = []
    quantiles = []
    for i in range(bins):
        quantiles.append(np.quantile(confidence, (i+0.5)/bins))
        mask = np.bitwise_and(confidence>np.quantile(confidence, i/bins), confidence<np.quantile(confidence, (i+1)/bins))
        accuracy.append(np.mean(np.equal(predicted_class[mask], target_class[mask])))
    data = pd.DataFrame({"quantiles": quantiles, "accuracy": accuracy})
    sns.relplot(data=data, x="quantiles", y="accuracy")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.plot([0,1], [0,1], 'k-')
    if not os.path.isdir("data/test"):
        os.makedirs("data/test")
    plt.savefig(f"data/test/confidence.png")

def show_interpretability_plot(preprocessing_filename, interpret_filename='interpret.csv', output_path='data/test', aggregation_fn=np.mean):
    """ Do calculations and create interpretability image as overlay on top of original whole slide image
    """
    #TODO: incorporate eval_scheme='x', assign_scheme='x',
    #TODO: cope with duplicates in bag_x and bag_y

    # Define a custom converter functions
    def parse_list(s):
        try:
            return ast.literal_eval(s)
        except (SyntaxError, ValueError):
            return s
    def parse_numpy_array(s):
        values_str = s.strip('[]')
        values = re.findall(r'[-+]?\d*\.\d+(?:e[-+]?\d+)?|\d+', values_str)
        values = [float(val) for val in values]  # Convert to float
        return np.array(values)

    def aggregate_arrays(arr_list, aggregation_fn=aggregation_fn):
        arr_list = np.array(arr_list)
        aggregated = aggregation_fn(arr_list, axis=0)
        return aggregated

    def overlay_png(input_path, output_path, tiles_list, values_list, tile_size=(224, 224), opacity=0.5):
        """
        Open and save a PNG file.

        Args:
            input_path (str): Path to the input PNG file.
            output_path (str): Path to save the output PNG file.
        """
        def create_overlay(overlay, tiles_list, tile_size, values_list, cmap=None):
            """
            Create a heatmap from input values and tile coordinates.

            Args:
                tiles_list: List of top-left corner coordinates of tiles (x, y).
                values_list: List of values to be color-coded.
                cmap (int, optional): OpenCV colormap to use. Default is seismic-like.
                tile_size (tuple, optional): Size of each tile in pixels (width, height). Default is (224, 224).
            """

            def calculate_log_ratios_and_scale(pairs):
                # Step 1: Calculate the logarithmic ratios
                # We add a small constant to avoid division by zero and log(0)
                small_constant = 1e-10
                log_ratios = [np.log((a + small_constant) / (b + small_constant)) for a, b in pairs]

                # Convert to a NumPy array for easier manipulation
                log_ratios = np.array(log_ratios)

                # Step 2: Find min and max values
                min_val = np.min(log_ratios)
                max_val = np.max(log_ratios)

                # Handle the special case where all values are the same (and hence min_val == max_val)
                if min_val == max_val:
                    return np.full_like(log_ratios, 128, dtype=np.uint8)

                # Step 3: Rescale to fit into the range [0, 255]
                # Center so that a log_ratio of 0 maps to 128
                scale_factor = max(abs(min_val), abs(max_val)) / 127.0
                scaled_log_ratios = np.round(log_ratios / scale_factor + 128).astype(np.uint8)

                return scaled_log_ratios

            def create_seismic_colormap():
                # Create a 256x1x3 array for storing RGB values
                colormap = np.zeros((256, 1, 3), dtype=np.uint8)

                # Create first half of colormap (from blue to white)
                for i in range(128):
                    colormap[i, 0, 0] = int(i * 2)  # Blue
                    colormap[i, 0, 1] = int(i * 2)  # Green
                    colormap[i, 0, 2] = 255  # Red

                # Create second half of colormap (from white to red)
                for i in range(128, 256):
                    colormap[i, 0, 0] = 255  # Blue
                    colormap[i, 0, 1] = int((255 - i) * 2)  # Green
                    colormap[i, 0, 2] = int((255 - i) * 2)  # Red

                return colormap
            if cmap is None:
                cmap = create_seismic_colormap()
            else:
                cmap

            # create metric for every value pair and rescale
            values_list = calculate_log_ratios_and_scale(values_list)

            # Map the input values to colors using a colormap
            colormap = cv2.applyColorMap(np.array(values_list, dtype=np.uint8), cmap)

            # draw rectangles
            for tile, color in zip(tiles_list, colormap):
                x, y = tile
                overlay[y:y+tile_size[1], x:x+tile_size[0]] = color

            return overlay

        image = cv2.imread(input_path)

        assert image is not None, "file could not be loaded"

        overlay = create_overlay(image.copy(), tiles_list, tile_size, values_list)

        cv2.addWeighted(overlay, opacity, image, 1-opacity, 0, image)  # Combine overlay with the original image
        cv2.imwrite(output_path, image)

    # load csvs
    df_preprocessing = pd.read_csv(preprocessing_filename, index_col=0)
    df_interpret = pd.read_csv(interpret_filename, index_col=0, converters={'predictions': parse_numpy_array})

    # parse string representation of list to list
    df_interpret['bag_x'] = df_interpret['bag_x'].apply(parse_list)

    #TODO: incorporate eval_scheme here: optionally find matches that belong together
    # modify df_interpret by selecting pairs of rows that share bag_y (if necessary)
    # then combining results, removing the original rows

    #TODO: incorporate assign_scheme here: so far only assigned to bag_x
    # introduce new column "concerned_tiles" indicating which tiles the results are to be assigned to
    df_interpret['concerned_tiles'] = df_interpret['bag_x']

    # Create a dictionary mapping tiles to values from df_interpret
    tile_to_value = {}
    for index, row in df_interpret.iterrows():
        for tile_path in row['concerned_tiles']:
            tile_to_value.setdefault(tile_path, []).append(row['predictions'])

    # generate output column (collect all predictions for every tile)
    df_preprocessing['predictions'] = df_preprocessing['slice_path'].apply(lambda name: tile_to_value.get(name, []))

    # aggregate results using aggregation_fn
    df_preprocessing['predictions_aggregated'] = df_preprocessing['predictions'].apply(lambda x: aggregate_arrays(x, aggregation_fn))

    # discard all rows with no predictions
    df_preprocessing = df_preprocessing[df_preprocessing['predictions_aggregated'].apply(lambda x: isinstance(x, np.ndarray))]
    df_preprocessing.to_csv("tmp.csv")

    # group by whole slide image and loop over
    grouped = df_preprocessing.groupby('image_path')

    # Initialize empty lists to store results
    tile_pos_list = []
    tile_size_list = []
    predictions_list = []

    # Iterate through groups
    for image_path, group in grouped:
        tile_pos = list(zip(group['y_coord'], group['x_coord']))
        tile_size = list(zip(group['y_size'], group['x_size']))
        predictions = group['predictions_aggregated'].tolist()

        tile_pos_list.append(tile_pos)
        tile_size_list.append(tile_size)
        predictions_list.append(predictions)

    # open whole slide image, create overlay of corresponding results from tiles, combine, save
    for i, image_path in enumerate(grouped.groups.keys()):
        overlay_png(image_path, os.path.join(output_path, os.path.basename(image_path)), tile_pos_list[i], predictions_list[i], tile_size_list[i][0], opacity=0.5)
