from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import pandas as pd
import os
from werkzeug.utils import secure_filename
from utils.data_processing import create_binary_datasets, save_to_arff, load_datasets, preprocess_datasets
from utils.model_evaluation import evaluate_models
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression  # Assicurati di importare LogisticRegression

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/uploads/'

@app.route('/')
def index():
    files_present = os.path.exists('data/converted') and any(fname.endswith('.arff') for fname in os.listdir('data/converted'))
    return render_template('index.html', files_present=files_present)

@app.route('/convert', methods=['POST'])
def convert():
    if 'files' not in request.files:
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
    save_to_arff(binary_datasets, [f'data/converted/file{i+1}.arff' for i in range(len(binary_datasets))])

    return jsonify({'message': 'Files converted successfully'})

@app.route('/results', methods=['POST'])
def results():
    datasets = load_datasets('data/converted')
    datasets = preprocess_datasets(datasets)  # Preprocessa i dataset per gestire i NaN
    binary_datasets = create_binary_datasets(datasets)
    binary_results = evaluate_models(binary_datasets)

    multi_class_dataset = pd.concat(datasets)
    X_multi = multi_class_dataset.drop(columns=['class'])
    y_multi = multi_class_dataset['class']
    multi_class_model = LogisticRegression(max_iter=500)  # Usare un modello più semplice per debugging
    skf = StratifiedKFold(n_splits=5)
    try:
        precision_multi = cross_val_score(multi_class_model, X_multi, y_multi, cv=skf, scoring='precision_weighted', error_score='raise').mean()
        recall_multi = cross_val_score(multi_class_model, X_multi, y_multi, cv=skf, scoring='recall_weighted', error_score='raise').mean()
    except ValueError as e:
        print(f"Error evaluating multi-class model: {e}")
        precision_multi, recall_multi = None, None

    multi_class_results = (precision_multi, recall_multi)

    # Assicurati che la directory dei risultati esista
    if not os.path.exists('data/results'):
        os.makedirs('data/results')

    # Esporta i risultati in un file CSV
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
