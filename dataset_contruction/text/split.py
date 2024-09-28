import pandas as pd


input_csv = 'path/benchmark/split/finalbench.csv'
df = pd.read_csv(input_csv)


for i, col_name in enumerate(df.columns, start=1):
    output_csv = f'{i}.csv'
    df[[col_name]].to_csv(output_csv, index=False, header=True)
    print(f'Saved column {col_name} to {output_csv}')
