import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score


def evaluate_models(datasets):
    results = []
    for dataset in datasets:
        X = dataset.drop(columns=['class'])
        y = dataset['class']

        if X.isnull().values.any():
            model = HistGradientBoostingClassifier(random_state=42)
        else:
            model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)

        skf = StratifiedKFold(n_splits=5)
        try:
            precision = cross_val_score(model, X, y, cv=skf, scoring='precision').mean()
            recall = cross_val_score(model, X, y, cv=skf, scoring='recall').mean()
        except ValueError:
            precision, recall = 0.0, 0.0

        results.append((precision, recall))
    return results


def evaluate_multi_class_model(dataset):
    X = dataset.drop(columns=['class'])
    y = dataset['class']

    model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
    skf = StratifiedKFold(n_splits=5)

    precision = cross_val_score(model, X, y, cv=skf, scoring='precision_weighted').mean()
    recall = cross_val_score(model, X, y, cv=skf, scoring='recall_weighted').mean()

    return precision, recall
