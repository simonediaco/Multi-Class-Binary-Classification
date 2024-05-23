import arff

def save_as_arff(dataframes, filenames):
    for df, filename in zip(dataframes, filenames):
        arff_data = {
            'description': '',
            'relation': 'relation_name',
            'attributes': [(col, 'REAL') for col in df.columns[:-1]] + [('class', ['0', '1'])],
            'data': df.values
        }
        with open(filename, 'w') as f:
            arff.dump(arff_data, f)
