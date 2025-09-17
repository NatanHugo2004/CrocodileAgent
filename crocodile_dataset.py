import kagglehub
import random
import pandas as pd
path = kagglehub.dataset_download("zadafiyabhrami/global-crocodile-species-dataset") + '\crocodile_dataset.csv'
pd = pd.read_csv(path)

lista_aleatoria = random.sample(range(0, len(pd)), 3)
common_names = list(pd['Common Name'])
common_names_dict = {pd.loc[i, "Common Name"]: pd.loc[i,"Notes"] for i in lista_aleatoria}
crocodile_test = common_names[lista_aleatoria[0]]
notes = common_names_dict[crocodile_test] 
