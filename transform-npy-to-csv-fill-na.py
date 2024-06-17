import numpy as np
import pandas as pd
import os

directory_path = './data/feature-vectors-npy/Validation/r21d/r2plus1d_18_16_kinetics'

data_rows = []
file_names = []

for file in os.listdir(directory_path):
    if file.endswith('.npy'):
        file_path = os.path.join(directory_path, file)
        data = np.load(file_path)
        flattened_data = data.flatten().tolist()
        data_rows.append(flattened_data)
        file_names.append(file)


df = pd.DataFrame(data_rows)

num_columns = df.shape[1]

headers = ['Feature_' + str(i+1) for i in range(num_columns)]
df.columns = headers

df['Filename'] = file_names

median_values = df[df.columns.difference(['Filename'])].median()
df[df.columns.difference(['Filename'])] = df[df.columns.difference(['Filename'])].fillna(median_values)

output_csv_path = './procesed-data-validation/full/r21.csv'
df.to_csv(output_csv_path, index=False, header=True) 

print(f'Datos guardados exitosamente en: {output_csv_path}')
