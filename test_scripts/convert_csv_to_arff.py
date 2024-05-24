import csv
import arff
import os

# Configurazione logging
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', filename='conversion.log')


def csv_to_arff(input_csv, output_arff, relation_name="binary_classification"):
    with open(input_csv, 'r') as fp:
        reader = csv.reader(fp)
        header = None
        data = []
        for row in reader:
            if header is None:
                header = row
            else:
                data.append(row)

    content = {
        'relation': relation_name,
        'attributes': [(n, 'NUMERIC') if n != 'class' else (n, ['0', '1']) for n in header],
        'data': data
    }

    with open(output_arff, 'w') as fp:
        arff.dump(content, fp)
    logging.info(f"Converted {input_csv} to {output_arff}")


def main():
    upload_folder = 'data/uploads'
    converted_folder = 'data/converted'

    if not os.path.exists(converted_folder):
        os.makedirs(converted_folder)

    csv_files = [f for f in os.listdir(upload_folder) if f.endswith('.csv')]
    for i, csv_file in enumerate(csv_files):
        input_csv = os.path.join(upload_folder, csv_file)
        output_arff = os.path.join(converted_folder, f'file{i + 1}.arff')
        csv_to_arff(input_csv, output_arff)


if __name__ == '__main__':
    main()
