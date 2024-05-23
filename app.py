from flask import Flask, render_template, request
import os
from utils.data_processing import read_csv_files, create_binary_datasets
from utils.arff_converter import save_as_arff

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate_datasets():
    file_paths = ['./data/1.csv', './data/2.csv', './data/3.csv']
    dataframes = read_csv_files(file_paths)
    binary_datasets = create_binary_datasets(dataframes)
    save_as_arff(binary_datasets, ['./data/generated_arffs/1.arff', './data/generated_arffs/2.arff', './data/generated_arffs/3.arff'])
    return "Datasets generated successfully!"


if __name__ == '__main__':
    app.run(debug=True)
