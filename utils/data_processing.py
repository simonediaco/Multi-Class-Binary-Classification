import pandas as pd
from scipy.io import arff
import os


def create_binary_datasets(datasets):
    binary_datasets = []
    for i in range(len(datasets)):
        positive_class = datasets[i].copy()
        negative_classes_list = [datasets[j] for j in range(len(datasets)) if j != i]
        negative_classes = pd.concat(negative_classes_list, ignore_index=True)

        positive_class['class'] = 1
        negative_classes['class'] = 0
        binary_dataset = pd.concat([positive_class, negative_classes], ignore_index=True)
        binary_dataset.fillna(0, inplace=True)  # Fill NaN values with 0

        binary_datasets.append(binary_dataset)
    return binary_datasets


def preprocess_datasets(datasets):
    for dataset in datasets:
        dataset.fillna(0, inplace=True)
    return datasets


def load_datasets(folder):
    arff_files = [f'{folder}/{file}' for file in os.listdir(folder) if file.endswith('.arff')]
    datasets = []
    for file in arff_files:
        data, meta = arff.loadarff(file)
        df = pd.DataFrame(data)
        df = df.apply(pd.to_numeric, errors='ignore')
        datasets.append(df)
    return datasets
