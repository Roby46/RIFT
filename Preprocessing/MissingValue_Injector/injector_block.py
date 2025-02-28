import random
import pandas as pd
import os
import csv
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def determine_column_type(col):
    if col.dtype == 'bool':
        return 'B'
    elif col.dtype == 'object' and col.str.len().max() == 1:
        return 'C'
    else:
        return 'D'
# Generate inviolable row indices
def generate_inviolable_indices(num_rows, n_inviolable):
    return set(random.sample(range(num_rows), min(n_inviolable, num_rows)))


# Check if file exists
def is_file_exist(file_path):
    return os.path.exists(file_path)

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

# Write a row to CSV
def write_row_to_csv(file_path, row_data):
    mode = 'a' if is_file_exist(file_path) else 'w'
    with open(file_path, mode, newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(row_data)


percentages = [10]
lengths = [20000]
columns = [130]
iterations = [1]
null_value = '?'
delimiter = ';'
max_malfunction_duration = 15

for length in lengths:
    for column_number in columns:
        dataset = 'S4-ADL5-MNAR'
        dataset = f"{dataset}_{length}_{column_number}"
        path_file = f'../../Datasets/Preprocessed_Datasets/{dataset}.csv'
        print(path_file)

        df = pd.read_csv(path_file, sep=delimiter)
        total_values = len(df.columns) * len(df)

        column_types = [determine_column_type(df[col]) for col in df.columns]

        for iteration in iterations:
            for p in percentages:
                df = pd.read_csv(path_file, sep=delimiter)
                mv_target = round(total_values * (p / 100))
                mv_count = 0



                missing_dataset_name = f'{dataset}_{mv_target}_{iteration}'


                column_types_data = column_types + [missing_dataset_name]
                column_types_output_path = f'../../Preprocessing/ColumnTypes/ColumnTypes.csv'
                write_unique_row_to_csv(column_types_output_path, column_types_data)

                header_data = df.columns.tolist() + [missing_dataset_name]
                header_output_path = f'../../Preprocessing/Headers/Headers.csv'
                write_unique_row_to_csv(header_output_path, header_data)


                path_initial_tuples_output = f'../../Preprocessing/Initial_Tuples/{dataset}/{missing_dataset_name}.csv'
                os.makedirs(os.path.dirname(path_initial_tuples_output), exist_ok=True)

                while mv_count < mv_target:
                    sensor_col = random.randint(0, column_number - 1)
                    start_time = random.randint(0, length - 1)
                    malfunction_duration = random.randint(1, max_malfunction_duration)

                    # Limit duration to avoid exceeding mv_target
                    remaining_mvs = mv_target - mv_count
                    malfunction_duration = min(malfunction_duration, remaining_mvs)

                    for i in range(malfunction_duration):
                        row_idx = start_time + i
                        if row_idx >= length:
                            break
                        if df.iat[row_idx, sensor_col] != null_value:
                            raw_data = [row_idx + 1, df.columns[sensor_col], df.iat[row_idx, sensor_col]]
                            write_row_to_csv(path_initial_tuples_output, raw_data)
                            df.iat[row_idx, sensor_col] = null_value
                            mv_count += 1
                            if mv_count >= mv_target:
                                break  # Stop if we reach the target missing values

                print(f"{mv_count} missing values injected for {missing_dataset_name}")
                count_question_marks = (df == "?").sum().sum()
                print(count_question_marks, "Missing values at the beginning")

                df_complete = df[~df.isin([null_value]).any(axis=1)]
                path_file_output = f'../../Datasets/Missing_Datasets/{dataset}/{missing_dataset_name}.csv'
                os.makedirs(os.path.dirname(path_file_output), exist_ok=True)
                df.to_csv(path_file_output, index=None, sep=";", header=False)

                path_file_output = f'../../Datasets/DOMINO_Datasets/{dataset}/{missing_dataset_name}.csv'
                os.makedirs(os.path.dirname(path_file_output), exist_ok=True)
                df_complete.to_csv(path_file_output, index=None, sep=";", header=False)
                print(len(df_complete))
