from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import pandas as pd
from werkzeug.utils import secure_filename
from utils.data_processing import create_binary_datasets, load_datasets, preprocess_datasets, combine_datasets, \
    preprocess_multi_class_dataset
from utils.model_evaluation import evaluate_models, evaluate_multi_class_model
from utils.csv_to_arff import csv_to_arff

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/uploads/'

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

@app.route('/')
def index():
    files_present = os.path.exists('data/converted') and any(fname.endswith('.arff') for fname in os.listdir('data/converted'))
    return render_template('index.html', files_present=files_present)

@app.route('/convert', methods=['POST'])
def convert():
    if 'files' not in request.files:
        return jsonify({'error': 'No files part in the request'}), 400

    files = request.files.getlist('files')
    ensure_directory_exists(app.config['UPLOAD_FOLDER'])
    for file in files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

    try:
        ensure_directory_exists('data/converted')
        csv_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.csv')]
        for i, csv_file in enumerate(csv_files):
            input_csv = os.path.join(app.config['UPLOAD_FOLDER'], csv_file)
            output_arff = os.path.join('data/converted', f'file{i+1}.arff')
            csv_to_arff(input_csv, output_arff)
        message = "Files converted successfully"
        message_type = "success"
    except ValueError as e:
        message = str(e)
        message_type = "danger"

    files_present = os.path.exists('data/converted') and any(fname.endswith('.arff') for fname in os.listdir('data/converted'))
    return render_template('index.html', files_present=files_present, message=message, message_type=message_type)

@app.route('/results', methods=['POST'])
def results():
    datasets = load_datasets('data/converted')
    datasets = preprocess_datasets(datasets)
    binary_datasets = create_binary_datasets(datasets)
    binary_results = evaluate_models(binary_datasets)

    combined_dataset = combine_datasets(datasets)
    combined_dataset = preprocess_multi_class_dataset(combined_dataset)
    multi_class_results = evaluate_multi_class_model(combined_dataset)

    ensure_directory_exists('data/results')

    results_df = pd.DataFrame(binary_results, columns=['Precision', 'Recall'])
    results_df.to_csv('data/results/binary_results.csv', index=False)

    multi_class_results_df = pd.DataFrame([multi_class_results], columns=['Precision', 'Recall'])
    multi_class_results_df.to_csv('data/results/multi_class_results.csv', index=False)

    return render_template('results.html', binary_results=binary_results, multi_class_results=multi_class_results)


@app.route('/data/results/<path:filename>')
def download_file(filename):
    return send_from_directory('data/results', filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
