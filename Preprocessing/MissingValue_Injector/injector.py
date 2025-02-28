import random
import pandas as pd
import os
import warnings
import csv


warnings.simplefilter(action='ignore', category=FutureWarning)
def is_row_in_file(file_path, row_data, delimiter=';'):
    if not os.path.exists(file_path):
        return False
    with open(file_path, 'r', newline='') as file:
        for line in file:
            if line.strip() == delimiter.join(row_data):
                return True
    return False

def write_unique_row_to_csv(file_path, row_data, delimiter=';'):
    if not is_row_in_file(file_path, row_data, delimiter):
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file, delimiter=delimiter)
            writer.writerow(row_data)

def determine_column_type(col):
    if col.dtype == 'bool':
        return 'B'
    elif col.dtype == 'object' and col.str.len().max() == 1:
        return 'C'
    else:
        return 'D'

def is_file_exist(file_path):
    return os.path.exists(file_path)

def write_row_to_csv(file_path, row_data):
    mode = 'a' if is_file_exist(file_path) else 'w'
    with open(file_path, mode, newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(row_data)

def generate_inviolable_indices(df_length, n):
    first_half = random.sample(range(0, df_length // 2), n)
    second_half = random.sample(range(df_length // 2, df_length), n)
    return set(first_half + second_half)

percentages = [5,10,20]
lengths=[3000]
columns=[13]
iterations = [1]
null_value = '?'
delimiter = ';'
dataset = 'Air'

for length in lengths:
    for column_number in columns:

        column_types = [""]
        dataset_complete = f"{dataset}_{length}_{column_number}"
        path_file = f'../../Datasets/Preprocessed_Datasets/{dataset_complete}.csv'

        print(path_file)

        df = pd.read_csv(path_file, sep=delimiter)
        totalValues = len(df.columns.tolist()) * len(df)
        print(len(df))

        print(totalValues)
        column_types = [determine_column_type(df[col]) for col in df.columns]

        for iteration in iterations:
            for p in percentages:
                df = pd.read_csv(path_file, sep=delimiter)
                MV_number = round(totalValues * (p / 100))
                setted_MVs = 0

                missing_dataset_name = f'{dataset_complete}_{MV_number}_{iteration}'

                column_types_data = column_types + [missing_dataset_name]
                column_types_output_path = f'../../Preprocessing/ColumnTypes/ColumnTypes.csv'
                write_unique_row_to_csv(column_types_output_path, column_types_data)

                header_data = df.columns.tolist() + [missing_dataset_name]
                header_output_path = f'../../Preprocessing/Headers/Headers.csv'
                write_unique_row_to_csv(header_output_path, header_data)

                path_initial_tuples_output = f'../../Preprocessing/Initial_Tuples/{dataset_complete}/{missing_dataset_name}.csv'
                directory = os.path.dirname(path_initial_tuples_output)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                while setted_MVs < MV_number:
                    riga = random.randint(0, len(df) - 1)
                    colonna = random.randint(0, len(df.columns.tolist()) - 1)
                    if df.iloc[riga, colonna] != null_value:
                        raw_data = [riga + 1, df.columns.tolist()[colonna], df.iloc[riga, colonna]]
                        write_row_to_csv(path_initial_tuples_output, raw_data)

                        df.iloc[riga, colonna] = null_value
                        setted_MVs += 1

                count_question_marks = (df == "?").sum().sum()
                print(count_question_marks, "MVs injected")

                df_complete = df[~df.isin([null_value]).any(axis=1)]

                path_file_output = f'../../Datasets/Missing_Datasets/{dataset_complete}/{missing_dataset_name}.csv'
                directory = os.path.dirname(path_file_output)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                df.to_csv(path_file_output, index=None, sep=";", header=False)

                path_file_output = f'../../RFDs/{dataset_complete}/{missing_dataset_name}/'
                directory = os.path.dirname(path_file_output)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                path_file_output = f'../../Datasets/DOMINO_Datasets/{dataset_complete}/{missing_dataset_name}.csv'
                directory = os.path.dirname(path_file_output)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                df_complete.to_csv(path_file_output, index=None, sep=";", header=False)



