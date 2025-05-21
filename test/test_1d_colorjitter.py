import torch
import torchvision.io
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from math import ceil
from utils.bagtransform import ColorJitter_1D

name_csv = 'data/all/data_preprocessing_all_x10_1vs23'
output_path = Path("data/test/test_1d_colojitter")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(name_csv, usecols=["slice_path"])
list_pos = [2, 20, 200, 2000, 20000]

# Data Augmentation - Train vs Test & Train
transform = ColorJitter_1D(jitter_intensity=0.25)

transf = 6

for pos in list_pos:
    slice_path = df.iloc[pos].slice_path

    # Load img data
    img_tensor = torchvision.io.read_image(slice_path).float() / 255

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.axis("off")
    ax1 = fig.add_subplot(2, ceil(transf / 2), 1)
    ax1.imshow(img_tensor[0], cmap='gray')

    for i in range(1, transf):
        # Apply transformations
        img_tensor = torchvision.io.read_image(slice_path).float() / 255
        img_tensor_t = transform(img_tensor)

        ax1 = fig.add_subplot(2, ceil(transf / 2), i + 1)
        ax1.imshow(img_tensor_t[0], cmap='gray')

    plt.savefig(Path(output_path, slice_path.split("/")[-1]))
    plt.close()
