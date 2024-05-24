import pandas as pd
import os

def load_and_verify_csvs(folder):
    csv_files = [f'{folder}/{file}' for file in os.listdir(folder) if file.endswith('.csv')]
    for file in csv_files:
        df = pd.read_csv(file)
        print(f'File: {file}')
        print(df.head())
        print(f'Number of NaNs:\n{df.isna().sum()}')
        print('\n')

# Percorso ai file CSV
folder_path = 'data/uploads'
load_and_verify_csvs(folder_path)
