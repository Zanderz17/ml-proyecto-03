import pandas as pd

csv_path1 = './data/feature-vectors-csv/bloque01/s3d/output.csv'
csv_path2 = './data/feature-vectors-csv/bloque02/s3d/output.csv'

df1 = pd.read_csv(csv_path1)
df2 = pd.read_csv(csv_path2)

df_concatenado = pd.concat([df1, df2], ignore_index=True)

output_csv_path = './procesed-data/full/s3d.csv'
df_concatenado.to_csv(output_csv_path, index=False)

print(f'El DataFrame concatenado tiene {len(df_concatenado)} filas.')
