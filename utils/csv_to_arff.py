import csv
import arff

def csv_to_arff(input_csv, output_arff):
    with open(input_csv, 'r') as fp:
        reader = csv.reader(fp)
        header = next(reader)
        data = [row for row in reader]

    content = {
        'relation': "binary_classification",
        'attributes': [(col, 'REAL') for col in header],
        'data': data
    }

    with open(output_arff, 'w') as fp:
        arff.dump(content, fp)
