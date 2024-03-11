import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from utils.bagmodel import BagModel
from utils.bagmodel import ConfidenceCrossEntropyLoss
from utils.modelinstance import ModelInstance
from utils.modelgovernance import ModelGovernance
from datetime import datetime
import sklearn.metrics as sm
import os
import ast
import torchvision
import gc


class Model:
    LIST_AVAILABLE_BAG_MODEL_PREP = ["resnet34", "resnet18", "resnet18_pathology"]
    LIST_AVAILABLE_BAG_MODEL_AFTER = ["default"]
    LIST_AVAILABLE_BAG_MODEL_MIDDLE = ["Mean", "Max", "MinMax", "MinMax_#", "SelfAttention"]
    LIST_AVAILABLE_OPTIMIZER = ["Adam"]
    LIST_AVAILABLE_LOSS_FUNCTION = ["CrossEntropyLoss", "ConfidenceCrossEntropyLoss"]

    def __init__(self, device, epochs_to_train=10, epochs_trained=0, bagmodel_prep="resnet34", bagmodel_after="default",
                 bagmodel_middle="Mean", optimizer_name="Adam", learning_rate=0.0001,
                 loss_function_name="CrossEntropyLoss", nb_input_channels=1, nb_classes=2, confidence=False, temperature=1):

        self.device = device
        self.epochs_to_train = epochs_to_train
        self.epochs_trained = epochs_trained
        self.bagmodel_prep = bagmodel_prep
        self.bagmodel_after = bagmodel_after
        self.bagmodel_middle = bagmodel_middle
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.loss_function_name = loss_function_name
        self.weights_file_name = None
        self.nb_input_channels = nb_input_channels
        self.nb_classes = nb_classes
        self.confidence = confidence # train extra confidence output
        self.temperature = temperature # parameter for temperature scaled confidence

        # Results
        self.confusion_matrix = None
        self.auc = None

        # Main parameters
        self.bagModel = None
        self.optimizer = None
        self.loss_function = None

        # create scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()

    def initialize(self):
        self.valid()
        # initialize model
        self.bagModel = BagModel(self.bagmodel_prep, self.bagmodel_after, self.bagmodel_middle,
                                 self.nb_input_channels, self.nb_classes, self.confidence).to(self.device).model

        if self.weights_file_name is not None:
            self.bagModel.load_state_dict(torch.load("Model Storage/Models/" + self.weights_file_name))

        # define optimizer
        if self.optimizer_name == "Adam":
            self.optimizer = Adam(self.bagModel.parameters(), lr=self.learning_rate)

        # define loss funciton
        if self.loss_function_name == "CrossEntropyLoss":
            # MyHingeLoss
            # Loss Function
            self.loss_function = nn.CrossEntropyLoss()
        if self.loss_function_name == "ConfidenceCrossEntropyLoss":
            self.loss_function = ConfidenceCrossEntropyLoss()

    def train(self, data):
        self.valid()
        print(" - - - - - - TRAIN - - - - - - ")

        n_total_steps = len(data)
        for epoch in range(self.epochs_trained, self.epochs_to_train):
            for i, (instance, ids, labels) in enumerate(data):
                # forward pass BAGS to the model
                with torch.cuda.amp.autocast():
                    predictions = self.bagModel((instance, ids))
                    loss = self.loss_function(predictions, labels)
                # reset gradients model.zero_grad() and optimizer.zero_grad() are the same
                # IF all your model parameters are in that optimizer
                self.optimizer.zero_grad()
                # backward pass
                self.scaler.scale(loss).backward()
                # update weights
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # print epochs / loss values
            print(f'Epoch [{epoch + 1}/{self.epochs_to_train}], Train Loss: {loss.item():.4f}')

        self.epochs_trained = self.epochs_to_train
        self.weights_file_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    def train_cv(self, bag_generator, i):
        self.valid()
        overall_predicitons = np.empty((0, 2), int)
        print(f" - - - - - - TRAIN - CV {i} - - - - - - ")
        # create fold-specific dataloader
        train_dl, test_dl = bag_generator.run(i)

        # re-initialize model
        self.weights_file_name = None
        self.initialize()
        self.epochs_trained = 0

        # Train model
        self.train(train_dl)

        # Test Model
        pred = self.test(test_dl, full_output=True)
        overall_predicitons = np.append(overall_predicitons, pred, axis=0)

        self.confusion_matrix = sm.confusion_matrix(overall_predicitons[:, 1], overall_predicitons[:, 0])


    def test(self, t_data, full_output=False, verbose=False):
        """ run model on test set.
        """
        self.valid()
        # get test_loader
        with torch.no_grad():
            print(" - - - - - - TEST - - - - - - ")

            predictions_raw_all = []
            labels_all = []
            # inference
            for instance, ids, labels in t_data:
                predictions_raw_all.append(self.bagModel((instance, ids)))
                labels_all.append(labels)

            # note: predictions may include values that do not correspond to classes (e.g. confidence)
            predictions_raw_all = torch.cat(predictions_raw_all)
            labels_all = torch.cat(labels_all)

            # apply softmax with temperature scaling
            predictions_raw_all[:, :self.nb_classes] = torch.softmax(predictions_raw_all[:, :self.nb_classes]/self.temperature, dim=1)

            # create predicted class tensor
            predicted_classes_all = torch.max(predictions_raw_all[:, :self.nb_classes], 1).indices

            # move all data to cpu/numpy
            predictions_raw_all = predictions_raw_all.cpu().numpy()
            predicted_classes_all = predicted_classes_all.cpu().numpy()
            labels_all = labels_all.cpu().numpy()

        self.confusion_matrix = sm.confusion_matrix(labels_all, predicted_classes_all)
        # AUC only applicable if 2 output classes:
        if predictions_raw_all.shape[1] == 2:
            self.auc = sm.roc_auc_score(labels_all, predictions_raw_all[:, 1])

        if verbose:
            print(f'Accuracy of the network: {sm.accuracy_score(labels_all, predicted_classes_all) * 100:.1f} %')
            print(f'Area under curve: {self.auc:.2f}')
            print(f'Classification report:\n {sm.classification_report(labels_all, predicted_classes_all)}')
            print(f'Confusion matrix:\n {self.confusion_matrix}')

        if full_output:
            # raw predictions, predicted labels and true labels
            return np.column_stack((predicted_classes_all, labels_all))
        return predicted_classes_all

    def predict(self, p_data, verbose=False):
        """ run model on predict set.
        """
        self.valid()
        # get test_loader
        with torch.no_grad():
            print(" - - - - - - PREDICT - - - - - - ")

            predictions_raw_all = []
            # inference
            for instance, ids, _ in p_data:
                predictions_raw_all.append(self.bagModel((instance, ids)))
                torch.cuda.empty_cache()
                gc.collect()

            # note: predictions may include values that do not correspond to classes (e.g. confidence)
            predictions_raw_all = torch.cat(predictions_raw_all)


            # apply softmax with temperature scaling
            predictions_raw_all[:, :self.nb_classes] = torch.softmax(predictions_raw_all[:, :self.nb_classes]/self.temperature, dim=1)

            # create predicted class tensor
            predicted_classes_all = torch.max(predictions_raw_all[:, :self.nb_classes], 1).indices

            # move all data to cpu/numpy
            predicted_classes_all = predicted_classes_all.cpu().numpy()

        return predicted_classes_all

    def interpret_test(self, t_data, interpret_filename='interpret.csv', verbose=False):
        """ special test function for interpretability bags
        """
        #TODO: apply temperature scaling like above if desired
        self.valid()
        with torch.no_grad():
            print(" - - - - - - INTERPRET - - - - - - ")

            predictions_raw_all = []
            bags_all = []
            labels_all = []

            # inference
            for instance, ids, labels in t_data:
                bags_all.append(instance)
                predictions_raw_all.append(self.bagModel((instance, ids)))
                labels_all.append(labels)

            # convert list to tensors
            predictions_raw_all = torch.cat(predictions_raw_all)
            labels_all = torch.cat(labels_all)
            # move all data to cpu/numpy
            predictions_raw_all = [row.cpu().numpy() for row in predictions_raw_all]
            labels_all = labels_all.cpu().numpy()

            df = pd.read_csv(interpret_filename, index_col='id')
            df_results = pd.DataFrame({'predictions': predictions_raw_all})
            assert len(df) == len(df_results), 'interpret baglist and results do not match in length'
            df = pd.concat([df, df_results], axis=1)
            df.to_csv(interpret_filename)

    def save_bagmodel_weights(self):
        if not os.path.isdir(ModelGovernance.PATH_MODELS):
            os.makedirs(ModelGovernance.PATH_MODELS)

        if self.weights_file_name is not None:
            torch.save(self.bagModel.state_dict(), ModelGovernance.PATH_MODELS + self.weights_file_name)

    def load_model_weights(self):
        if self.weights_file_name is not None:
            device = torch.device('cpu')
            self.bagModel = BagModel(self.bagmodel_prep, self.bagmodel_after, self.bagmodel_middle).to(
                self.device).model
            self.bagModel.load_state_dict(torch.load(ModelGovernance.PATH_MODELS + self.weights_file_name,
                                                     map_location=device))

    def get_attributes(self):
        # Attribute dict from Model
        return {'epochs_to_train': self.epochs_to_train, 'epochs_trained': self.epochs_trained, 'bagmodel prep': self.bagmodel_prep, 'bagmodel after': self.bagmodel_after,
                'bagmodel middle': self.bagmodel_middle, 'optimizer name': self.optimizer_name,
                'learning rate': self.learning_rate, 'loss function name': self.loss_function_name,
                'confidence output': self.confidence, 'confusion matrix': self.confusion_matrix.tolist(),
                'auc': self.auc, 'weights file name': self.weights_file_name}

    def load_attributes(self, dict_attributes):
        self.epochs_to_train = dict_attributes['epochs_to_train']
        self.epochs_trained = dict_attributes['epochs_trained']
        self.bagmodel_prep = dict_attributes['bagmodel prep']
        self.bagmodel_after = dict_attributes['bagmodel after']
        self.bagmodel_middle = dict_attributes['bagmodel middle']
        self.optimizer_name = dict_attributes['optimizer name']
        self.learning_rate = dict_attributes['learning rate']
        self.loss_function_name = dict_attributes['loss function name']
        self.confidence = dict_attributes['confidence output']
        self.confusion_matrix = dict_attributes['confusion matrix']
        self.auc = dict_attributes['auc'] # this is useless
        self.weights_file_name = dict_attributes['weights file name']

        self.confusion_matrix = np.array(ast.literal_eval(self.confusion_matrix))

        self.valid()

    def valid(self):

        valid = True
        problems = []

        if not isinstance(self.epochs_to_train, int) or self.epochs_to_train < 1:
            valid = False
            problems += ["Epochs class with problems"]

        if self.bagmodel_prep is None or not isinstance(self.bagmodel_prep, str) or \
                self.bagmodel_prep not in Model.LIST_AVAILABLE_BAG_MODEL_PREP:
            valid = False
            problems += ["Bag model prep not valid"]

        if self.bagmodel_after is None or not isinstance(self.bagmodel_after, str) or \
                self.bagmodel_after not in Model.LIST_AVAILABLE_BAG_MODEL_AFTER:
            valid = False
            problems += ["Bag model after not valid"]

        if self.bagmodel_middle is None or not isinstance(self.bagmodel_middle, str) or \
                not (self.bagmodel_middle in Model.LIST_AVAILABLE_BAG_MODEL_MIDDLE or
                     (self.bagmodel_middle.startswith("MinMax_") and int(self.bagmodel_middle.split("_")[1]))):
            valid = False
            problems += ["Bag model middle not valid"]

        if self.optimizer_name is None or not isinstance(self.optimizer_name, str) or \
                self.optimizer_name not in Model.LIST_AVAILABLE_OPTIMIZER:
            valid = False
            problems += ["Optimizer not valid"]

        if self.loss_function_name is None or not isinstance(self.loss_function_name, str) or \
                self.loss_function_name not in Model.LIST_AVAILABLE_LOSS_FUNCTION:
            valid = False
            problems += ["Loss function not valid"]

        if not isinstance(self.learning_rate, float) or self.learning_rate <= 0:
            valid = False
            problems += ["Learning rate with problems"]

        if not (self.confusion_matrix is None or (isinstance(self.confusion_matrix, np.ndarray) and
                                                  all([(isinstance(elem, np.integer) and elem >= 0)
                                                       for elem in self.confusion_matrix.flatten()]))):
            valid = False
            problems += ["Confusion Matrix with problems"]

        if not (self.weights_file_name is None or isinstance(self.weights_file_name, str)):
            valid = False
            problems += ["Weights file name with problems"]

        if not valid:
            print("MODEL - VALIDATION PROBLEM:")
            for problem in problems:
                print("\n" + problem)

        return valid
