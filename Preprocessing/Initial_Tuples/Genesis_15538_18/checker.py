import pandas as pd


nome_dataset = 'Genesis_15538_18'
nome_dataset_MV = f'{nome_dataset}_55937_1'

# Leggi il file degli header riga per riga
headers = []
with open('../../Headers/Headers.csv', 'r') as f:
    for line in f:
        valori = line.strip().split(';')
        headers.append(valori)

# Trova l'header specifico in base al nome del dataset
colonne = None
for header in headers:
    if header[-1] == nome_dataset_MV:  # Ultimo valore è il nome del dataset
        colonne = [col for col in header[:-1] if col]  # Escludi l'ultimo elemento e rimuovi eventuali valori vuoti
        break

if colonne is None:
    raise ValueError(f"Header per il dataset '{nome_dataset_MV}' non trovato nel file header.")

# Percorsi ai file
file_risultati = f"{nome_dataset_MV}_temp.csv"
file_dataset = f"../../../Datasets/Missing_Datasets/{nome_dataset}/{nome_dataset_MV}.csv"

# Caricare i file CSV
risultati = pd.read_csv(file_risultati, sep=";")
dataset = pd.read_csv(file_dataset, sep=';', names=colonne)


count_question_marks = (dataset == "?").sum().sum()
print(count_question_marks, "Missing values at the beginning")

for _, row in risultati.iterrows():
    indice = row['riga'] - 1  # Convertire in indice zero-based
    attributo = row['nome attributo']
    valore_imputato = row['valore']

    # Verificare se esiste un ? nel dataset originale
    if dataset.at[indice, attributo] == "?":
        # Rimpiazzare con il valore imputato
        pass
    else:
        # Stampare il problema se non si trova un '?'
        print(f"Problema alla riga {row['riga']}, colonna '{attributo}': "
              f"trovato '{dataset.at[indice, attributo]}' invece di '?'.")

