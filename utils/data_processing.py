import pandas as pd
import os


def read_csv_files(file_paths):
    dataframes = [pd.read_csv(file_path) for file_path in file_paths]
    return dataframes


def create_binary_datasets(dataframes):
    binary_datasets = []
    for i, df in enumerate(dataframes):
        positive_class = df.copy()
        negative_class = pd.concat([dataframes[j] for j in range(len(dataframes)) if j != i])
        negative_class['class'] = 0
        positive_class['class'] = 1
        binary_dataset = pd.concat([positive_class, negative_class])
        binary_datasets.append(binary_dataset)
    return binary_datasets


def split_data(X, y, test_size=0.3, random_state=42):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
