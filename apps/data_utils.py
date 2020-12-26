import pnadas as pd
import csv
import numpy as np
import datetime


def load_csv(filepath):
    """
    Loads csv file and returns a dataframe.
    """
    data_rows = []
    with open(filepath, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for lines in csvreader:
            data_rows.append(lines)
    data = pd.DataFrame(data_rows[1:], columns=data_rows[0])
    return data


def normalize(data):
    """
    Normalizes values of provided data columns by their max value
    and saves to a file. It is meant to overwrite on files from previous time
    period.
    """
    data_ = data.copy()
    for var in data_.columns:
        max_val = data[var].max()
        data_.loc[:, var] = data[var] / max_val
        np.save(os.path.join(SERVE_DIR, 'max_{}.npy'.format(var)), max_val)

    return data_


def mask_status(data, column, mask_dict):
    """
    Based on the "reopen" status types from NYT dataset, the status column
    is returned as integer representations.
    """
    mask = 1
    masked = data[column].apply(lambda x: mask if x in mask_dict[mask]
    else (mask + 1 if x in mask_dict[mask + 1]
          else mask + 2))

    return masked


def extract_cols(data, extract_str):
    """
    he function returns all columns with the given string
    in the column name.
    """
    return data[[col for col in data.columns if extract_str in col]]


def count_string_values(data, exclude_parser=';', val_len=3):
    """
    If specified criteria is met, a word is converted into an integer.
    """
    exp = exclude_parser
    return data.applymap(lambda x: len(','.join(str(x).lower().split(exp))
                                       .split(',')) if len(str(x)) > val_len else 0)


def serialize_dummy(arr):
    """
    Extracts and returns the max value among each dummy record.
    """
    out_list = []
    for row in arr:
        out_list.extend([int(i + 1) for i in range(len(row)) if row[i] == row.max(0)])
    return np.array(out_list)
