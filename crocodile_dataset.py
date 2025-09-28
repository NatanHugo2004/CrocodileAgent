import kagglehub
import random
import pandas as pd
path = kagglehub.dataset_download("zadafiyabhrami/global-crocodile-species-dataset") + '/crocodile_dataset.csv'
pd = pd.read_csv(path)

lista_aleatoria = random.sample(range(0, len(pd)), 3)
common_names = list(pd['Common Name'])
common_names_dict_notes = {row["Common Name"]: row["Notes"] for _, row in pd.iterrows()}
common_names_dict_country = {row["Common Name"]: row["Country/Region"] for _, row in pd.iterrows()}
common_names_dict_weight = {row["Common Name"]: row["Observed Weight (kg)"] for _, row in pd.iterrows()}
common_names_dict_habitat = {row["Common Name"]: row["Habitat Type"] for _, row in pd.iterrows()}
common_names_dict_conservation = {row["Common Name"]: row["Conservation Status"] for _, row in pd.iterrows()}
