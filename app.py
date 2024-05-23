from flask import Flask, render_template, request
import os
import logging
from utils.data_processing import read_csv_files, create_binary_datasets
from utils.arff_converter import save_as_arff
from models.binary_classification import evaluate_model
from models.multi_class_classification import evaluate_multi_class_model
import pandas as pd

app = Flask(__name__)

# Configura il logging
logging.basicConfig(level=logging.DEBUG)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate_datasets():
    try:
        logging.info("Generating datasets")
        file_paths = ['./data/1.csv', './data/2.csv', './data/3.csv']
        dataframes = read_csv_files(file_paths)
        binary_datasets = create_binary_datasets(dataframes)

        # Crea la cartella se non esiste
        output_dir = './data/generated_arffs'
        os.makedirs(output_dir, exist_ok=True)

        save_as_arff(binary_datasets, [f'{output_dir}/1.arff', f'{output_dir}/2.arff', f'{output_dir}/3.arff'])
        logging.info("Datasets generated successfully")
        return "Datasets generated successfully!"
    except Exception as e:
        logging.exception("Error generating datasets")
        return str(e), 500


@app.route('/train_binary', methods=['POST'])
def train_binary_models():
    try:
        logging.info("Starting binary model training...")
        file_paths = ['./data/1.csv', './data/2.csv', './data/3.csv']
        dataframes = read_csv_files(file_paths)
        binary_datasets = create_binary_datasets(dataframes)
        results = []
        for i, dataset in enumerate(binary_datasets):
            logging.debug(f"Training model {i + 1} with dataset")
            X = dataset.iloc[:, :-1].values
            y = dataset.iloc[:, -1].values
            logging.debug(f"Shape of X: {X.shape}, Shape of y: {y.shape}")
            precision, recall = evaluate_model(X, y)
            results.append((precision, recall))
        logging.info("Binary models evaluated successfully")
        return f"Binary Models Evaluated: {results}"
    except Exception as e:
        logging.exception("Error training binary models")
        return str(e), 500


@app.route('/train_multi_class', methods=['POST'])
def train_multi_class_model():
    try:
        logging.info("Starting multi-class model training...")
        file_paths = ['./data/1.csv', './data/2.csv', './data/3.csv']
        dataframes = read_csv_files(file_paths)
        combined_df = pd.concat(dataframes)
        X = combined_df.iloc[:, :-1].values
        y = combined_df.iloc[:, -1].values
        num_classes = len(set(y))
        accuracy, precision, recall = evaluate_multi_class_model(X, y, num_classes)
        logging.info("Multi-class model evaluated successfully")
        return f"Multi-Class Model Evaluated: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}"
    except Exception as e:
        logging.exception("Error training multi-class model")
        return str(e), 500


if __name__ == '__main__':
    app.run(debug=True)
