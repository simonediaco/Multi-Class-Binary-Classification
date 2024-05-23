import pandas as pd
import os
import arff
from sklearn.impute import SimpleImputer

def create_binary_datasets(datasets):
    binary_datasets = []
    for i in range(len(datasets)):
        positive_class = datasets[i].copy()
        negative_classes = pd.concat([datasets[j] for j in range(len(datasets)) if j != i]).copy()
        positive_class['class'] = 1
        negative_classes['class'] = 0
        binary_dataset = pd.concat([positive_class, negative_classes])
        binary_datasets.append(binary_dataset)
    return binary_datasets

def save_to_arff(datasets, filenames):
    for dataset, filename in zip(datasets, filenames):
        arff_data = {
            'description': '',
            'relation': 'binary_classification',
            'attributes': [(col, 'REAL') for col in dataset.columns],
            'data': dataset.values.tolist()
        }
        with open(filename, 'w') as f:
            arff.dump(arff_data, f)

def load_datasets(folder):
    arff_files = [f'{folder}/{file}' for file in os.listdir(folder) if file.endswith('.arff')]
    datasets = []
    for file in arff_files:
        data = arff.load(open(file, 'r'))
        df = pd.DataFrame(data['data'], columns=[attr[0] for attr in data['attributes']])
        datasets.append(df)
    return datasets

def preprocess_datasets(datasets):
    imputer = SimpleImputer(strategy='mean')
    for i in range(len(datasets)):
        datasets[i] = pd.DataFrame(imputer.fit_transform(datasets[i]), columns=datasets[i].columns)
    return datasets
