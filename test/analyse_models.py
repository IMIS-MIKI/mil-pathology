import pandas as pd
import numpy as np
import sklearn.metrics as sk
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import ast

path_models = "Model Storage/models.csv"

variables = ["PREPROCESSING_valid background perc", "GENERATOR_bags per class", "GENERATOR_slices per bag",
             "MODEL_bagmodel middle"]

df_models = pd.read_csv(path_models, sep=";", usecols=variables + ["OTHER_comment", "MODEL_confusion matrix"])
df_models[["accuracy"]] = 0
df_models[["f1_score"]] = 0
df_models[["bags_per_class"]] = 0

# Lines to skip
# 0 -> test one slice only
# 1 - 9 -> Normal CNN
# 47-61 - > Remove ConfidenceCrossEntropyLoss
# 73-83 -> Cross validation
# 84 -> final run
df_models = df_models.drop(index=[0,
                                  1, 2, 3, 4, 5, 6, 7, 8, 9,
                                  47, 49, 51, 53, 55, 57, 59, 61,
                                  73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
                                  84])

def generate_vectors(conf_mat):
    true_1, false_1, false_23, true_23 = conf_mat.flatten()
    true_label = (true_1 + false_23) * [1] + (false_1 + true_23) * [0]
    pred_label = true_1 * [1] + false_23 * [0] + false_1 * [1] + true_23 * [0]
    return true_label, pred_label


for index, row in df_models.iterrows():
    conf_mat = np.array(ast.literal_eval(row["MODEL_confusion matrix"]))
    true_label, pred_label = generate_vectors(conf_mat)
    df_models.loc[index, "accuracy"] = sk.accuracy_score(true_label, pred_label)
    df_models.loc[index, "f1_score"] = sk.f1_score(true_label, pred_label)

    # print(row["OTHER_comment"])
    # print(classification_report(true_label, pred_label))

df_models = df_models.sort_values(by="accuracy")


for var in variables:
    cols = [var, "accuracy"]
    # options = df_models[var].drop_duplicates().tolist()
    df_aux = df_models[cols]
    df_aux.columns = cols
    fig = plt.figure()
    fig.suptitle(var)
    df_aux.groupby(var).boxplot(subplots=False, column="accuracy")
    plt.show()

# string = ""
# for i in range(10):
#     string += str(i) + ", "