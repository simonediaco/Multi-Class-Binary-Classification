import pandas as pd

def read_csv_files(file_paths):
    return [pd.read_csv(file_path) for file_path in file_paths]

def create_binary_datasets(dataframes):
    binary_datasets = []
    for i, df in enumerate(dataframes):
        positive_class = df.copy()
        negative_class = pd.concat([dataframes[j] for j in range(len(dataframes)) if j != i])
        binary_dataset = pd.concat([positive_class, negative_class])
        binary_dataset['class'] = [1] * len(positive_class) + [0] * len(negative_class)
        binary_datasets.append(binary_dataset)
    return binary_datasets
