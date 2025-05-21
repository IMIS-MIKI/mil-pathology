from utils.model import Model
from utils.preprocessing import PreProcessing
from utils.baggenerator import BagGenerator
from utils.bagtransform import BagTransform
from utils.modelinstance import ModelInstance
from utils.modelgovernance import ModelGovernance
from pathlib import Path
import gc
# import argparse
import torch
import itertools

# preprocessing parameters
input_path = Path("")
output_path = Path("")
name_csv = ''
train_test_split = 0.8

# define image dye type. normal="normal", kongo-red="fluorescent"
image_type = "fluorescent"
normalize_globally = True  # with training set statistics
save_model = True

# ##### Parameters to evaluate ######
list_bags_per_class = [10000]
list_slices_per_bag = [15]
list_num_epochs = [80]  # args.epochs ...
list_batch_size = [5]
list_learning_rate = [0.00001]
list_perc_background = [0.5]
list_bag_model_middle = ["Mean"]
list_loss_function = ["CrossEntropyLoss"]

cv_folds = 10

##########################
def run_full_process(bags_per_class, slices_per_bag, num_epochs, batch_size, learning_rate, perc_background,
                     bag_model_middle, loss_function, save_model):
    torch.cuda.empty_cache()
    gc.collect()

    current_name_csv = name_csv + "_" + str(perc_background)

    if cv_folds:
        current_name_csv = current_name_csv + "_" + str(cv_folds)
    current_output_path = output_path / str(perc_background)

    # Pre-Processing. Train/Test splitting happens here. maybe reconsider
    preprocessing = PreProcessing(name_csv=current_name_csv, input_path=input_path, output_path=current_output_path,
                                  split_ratio=train_test_split, valid_background_perc=perc_background, cv_folds=10,
                                  verbose=True, image_type=image_type)
    preprocessing.run()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get number of channels from input images & name of the classes
    nb_input_channels = preprocessing.get_nb_input_channels()
    classes = preprocessing.get_classes_name()
    nb_classes = len(classes)

    # Get train set statistics to use for normalization via transforms below
    train_stats = preprocessing.get_train_set_statistics()

    # Data Augmentation - Train vs Test & Train
    transforms_train = BagTransform(all_orientations=True, color_jitter=True,
                                    normalize=train_stats if normalize_globally else None,
                                    nb_input_channels=nb_input_channels)
    transforms_train.autobuild()

    transforms_test = BagTransform(normalize=train_stats if normalize_globally else None)
    transforms_test.autobuild()

    # Load Data from baggenerator.py
    bag_generator = BagGenerator(device, transforms_train, transforms_test, classes=classes, name_csv=current_name_csv,
                                 bags_per_class=bags_per_class, slices_per_bag=slices_per_bag, batch_size=batch_size,
                                 verbose=True)
    train_dl, test_dl = bag_generator.run()

    # Generate Model
    model = Model(device, num_epochs, bagmodel_middle=bag_model_middle, loss_function_name=loss_function,
                  nb_input_channels=nb_input_channels, learning_rate=learning_rate)
    model.initialize()

    # Train model and store predictions
    for i in range(preprocessing.get_attributes()["cv folds"]):
        model.train_cv(bag_generator, i)

        # Save Data
        if save_model:
            model_governance = ModelGovernance()
            model_instance = ModelInstance()
            comment = f"Cross Validation - 1st class vs classes 2&3 with both batches and bags_per_class={bags_per_class} " \
                      f"slices_per_bag={slices_per_bag} num_epochs={num_epochs} batch_size={batch_size} " \
                      f"learning_rate={learning_rate} perc_background = {perc_background} " \
                      f"bag_model_middle={bag_model_middle} loss_function={loss_function} cv_folds={cv_folds}"
            model_instance.add_info(preprocessing.get_attributes(), transforms_train.get_attributes(),
                                    bag_generator.get_attributes(), model.get_attributes(), comment)
            model_governance.add_instance(model_instance)
            # model.save_bagmodel_weights()


# Iterate over all combinations of parameters
param_combinations = list(itertools.product(list_bags_per_class, list_slices_per_bag, list_num_epochs, list_batch_size,
                                            list_learning_rate, list_perc_background, list_bag_model_middle,
                                            list_loss_function))

for comb in param_combinations:
    bags_per_class, slices_per_bag, num_epochs, batch_size, learning_rate, perc_background, bag_model_middle, \
        loss_function = comb
    # break
    print(comb)
    run_full_process(bags_per_class, slices_per_bag, num_epochs, batch_size, learning_rate, perc_background,
                     bag_model_middle, loss_function, save_model)
