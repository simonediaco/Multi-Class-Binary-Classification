from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import os
from werkzeug.utils import secure_filename
from utils.data_processing import create_binary_datasets, save_to_arff, load_datasets, preprocess_datasets
from utils.model_evaluation import evaluate_models
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier  # Assicurati di importare RandomForestClassifier
import logging

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/uploads/'

# Creazione della cartella di log se non esiste
if not os.path.exists('logs'):
    os.makedirs('logs')

# Configurazione logging
debug_log = logging.FileHandler('logs/debug.log')
error_log = logging.FileHandler('logs/error.log')

debug_log.setLevel(logging.DEBUG)
error_log.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
debug_log.setFormatter(formatter)
error_log.setFormatter(formatter)

app.logger.addHandler(debug_log)
app.logger.addHandler(error_log)
app.logger.setLevel(logging.DEBUG)


@app.route('/')
def index():
    files_present = os.path.exists('data/converted') and any(
        fname.endswith('.arff') for fname in os.listdir('data/converted'))
    return render_template('index.html', files_present=files_present)


@app.route('/convert', methods=['POST'])
def convert():
    if 'files' not in request.files:
        app.logger.error('No files part in the request')
        return jsonify({'error': 'No files part in the request'}), 400

    files = request.files.getlist('files')
    datasets = []
    for file in files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        datasets.append(pd.read_csv(file_path))

    binary_datasets = create_binary_datasets(datasets)
    if not os.path.exists('data/converted'):
        os.makedirs('data/converted')
    save_to_arff(binary_datasets, [f'data/converted/file{i + 1}.arff' for i in range(len(binary_datasets))])

    app.logger.info('Files converted successfully')
    return jsonify({'message': 'Files converted successfully'})


@app.route('/results', methods=['POST'])
def results():
    datasets = load_datasets('data/converted')
    datasets = preprocess_datasets(datasets)
    binary_datasets = create_binary_datasets(datasets)

    # Logging delle distribuzioni delle classi binarie
    for i, dataset in enumerate(binary_datasets):
        app.logger.info(f"Dataset {i + 1} - Binary class distribution:\n{dataset['class'].value_counts()}")

    binary_results = evaluate_models(binary_datasets)

    multi_class_dataset = pd.concat(datasets)
    X_multi = multi_class_dataset.drop(columns=['class'])
    y_multi = multi_class_dataset['class']
    multi_class_model = RandomForestClassifier(n_estimators=100,
                                               random_state=42)  # Usare un modello più semplice per debugging
    skf = StratifiedKFold(n_splits=5)
    try:
        precision_multi = cross_val_score(multi_class_model, X_multi, y_multi, cv=skf, scoring='precision_weighted',
                                          error_score='raise').mean()
        recall_multi = cross_val_score(multi_class_model, X_multi, y_multi, cv=skf, scoring='recall_weighted',
                                       error_score='raise').mean()

        app.logger.info(f"Multi-class model - Precision: {precision_multi}, Recall: {recall_multi}")
    except ValueError as e:
        app.logger.error(f"Error evaluating multi-class model: {e}")
        precision_multi, recall_multi = None, None

    multi_class_results = (precision_multi, recall_multi)

    if not os.path.exists('data/results'):
        os.makedirs('data/results')

    results_df = pd.DataFrame(binary_results, columns=['Precision', 'Recall'])
    results_df.to_csv('data/results/binary_results.csv', index=False)

    multi_class_df = pd.DataFrame([multi_class_results], columns=['Precision', 'Recall'])
    multi_class_df.to_csv('data/results/multi_class_results.csv', index=False)

    return render_template('results.html', binary_results=binary_results, multi_class_results=multi_class_results)


@app.route('/data/results/<path:filename>')
def download_file(filename):
    return send_from_directory('data/results', filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
