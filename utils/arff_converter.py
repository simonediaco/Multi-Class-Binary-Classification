import pandas as pd
from scipy.io import arff


def save_as_arff(datasets, file_paths):
    for dataset, file_path in zip(datasets, file_paths):
        arff_data = {
            'description': '',
            'relation': 'binary_classification',
            'attributes': [(col, 'REAL') for col in dataset.columns[:-1]] + [('class', 'INTEGER')],
            'data': dataset.values,
        }
        with open(file_path, 'w') as f:
            arff.dump(arff_data, f)
