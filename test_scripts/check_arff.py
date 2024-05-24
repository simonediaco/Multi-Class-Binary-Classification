import pandas as pd
from scipy.io import arff

def load_arff(file_path):
    data = arff.loadarff(file_path)
    df = pd.DataFrame(data[0])
    return df

# Percorso ai file ARFF
arff_files = ['data/converted/file1.arff', 'data/converted/file2.arff', 'data/converted/file3.arff']

# Caricamento e verifica dei file ARFF
for file in arff_files:
    df = load_arff(file)
    print(f'File: {file}')
    print(df.head())
    print(f'Number of NaNs:\n{df.isna().sum()}')
    print('\n')
