import pandas as pd
from scipy.io import arff

# Carica il file CSV
csv_path = 'data/uploads/1.csv'
csv_data = pd.read_csv(csv_path)

# Carica il file ARFF
arff_path = 'data/converted/file1.arff'
arff_data, arff_meta = arff.loadarff(arff_path)
arff_df = pd.DataFrame(arff_data)

# Visualizza le prime righe e le informazioni di entrambi i file
print("CSV Data Info:")
print(csv_data.info())
print(csv_data.head())

print("\nARFF Data Info:")
print(arff_df.info())
print(arff_df.head())
