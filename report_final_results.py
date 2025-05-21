import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import LabelBinarizer


# Load the CSV file
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


# Compute AUC and F1 Score
def evaluate_model(df):
    y_true = df['TrueAmyloidType']
    y_pred = df['type']

    # Binarizing labels
    lb = LabelBinarizer()
    y_true_bin = lb.fit_transform(y_true)
    y_pred_bin = lb.transform(y_pred)

    # Compute AUC
    auc_score = roc_auc_score(y_true_bin, y_pred_bin, average='macro', multi_class='ovr')

    # Compute F1 Score
    f1 = f1_score(y_true, y_pred, average='macro')

    return auc_score, f1


if __name__ == "__main__":
    file_path = "data.csv"  # Change to your actual CSV file path
    df = load_data(file_path)
    auc, f1 = evaluate_model(df)

    print(f"AUC Score: {auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
