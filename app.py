from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neural_network import MLPClassifier

app = Flask(__name__)


def load_datasets():
    csv_files = ['data/1.csv', 'data/2.csv', 'data/3.csv']
    datasets = [pd.read_csv(file) for file in csv_files]
    return datasets


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


def evaluate_models(datasets):
    results = []
    for dataset in datasets:
        X = dataset.drop(columns=['class'])
        y = dataset['class']
        model = MLPClassifier(max_iter=500)
        skf = StratifiedKFold(n_splits=5)
        precision_scores = cross_val_score(model, X, y, cv=skf, scoring='precision')
        recall_scores = cross_val_score(model, X, y, cv=skf, scoring='recall')
        precision = precision_scores.mean()
        recall = recall_scores.mean()
        results.append((precision, recall))
    return results


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/results', methods=['POST'])
def results():
    datasets = load_datasets()
    binary_datasets = create_binary_datasets(datasets)
    binary_results = evaluate_models(binary_datasets)

    multi_class_dataset = pd.concat(datasets)
    X_multi = multi_class_dataset.drop(columns=['class'])
    y_multi = multi_class_dataset['class']
    multi_class_model = MLPClassifier(max_iter=500)
    skf = StratifiedKFold(n_splits=5)
    precision_multi = cross_val_score(multi_class_model, X_multi, y_multi, cv=skf, scoring='precision_weighted').mean()
    recall_multi = cross_val_score(multi_class_model, X_multi, y_multi, cv=skf, scoring='recall_weighted').mean()

    multi_class_results = (precision_multi, recall_multi)

    return render_template('results.html', binary_results=binary_results, multi_class_results=multi_class_results)


if __name__ == '__main__':
    app.run(debug=True)
