import kagglehub
import random
import pandas as pd
path = kagglehub.dataset_download("zadafiyabhrami/global-crocodile-species-dataset") + '/crocodile_dataset.csv'
pd = pd.read_csv(path)

crocodilos = open("./data/crocodilos.txt", "w")
crocodilos_set = set()
for _, row in pd.iterrows():
    mensagem = f"The crocodile {row['Common Name']} has the scientific name {row['Scientific Name']}, habitat: {row['Habitat Type']}, found in {row['Country/Region']}. It has an observed weight of {row['Observed Weight (kg)']} kg and a length of {row['Observed Length (m)']} m. Conservation status: {row['Conservation Status']}. Notes: {row['Notes']}"
    crocodilos_set.add(mensagem)  # Adiciona ao set (duplicatas serão ignoradas)

# Escreve os dados únicos no arquivo
with open("./data/crocodilos.txt", "w") as crocodilos:
    for i, mensagem in enumerate(crocodilos_set):
        if i >= 200:  # Limita a escrita a 200 mensagens
            break
        crocodilos.write(mensagem + "\n")