import pandas as pd
from pathlib import Path

df_int = pd.read_csv(Path(""))
df_fin = pd.read_csv(Path(""))

df_int["type_int"] = df_int["type"]
df_fin["type_fin"] = df_fin["type"]

df_int = df_int[['image', 'type_int']]
df_fin = df_fin[['image', 'type_fin']]

df_grouped = df_fin.merge(df_int, how="left", on="image")

df_grouped[df_grouped["type_int"] != df_grouped["type_fin"]]

#

df = pd.read_csv(Path("data/TC1/png/data_preprocessing_all_x10_1vs23_0.5"))

