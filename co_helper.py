import os
import numpy as np
import pandas as pd


def get_filename_list(file_dir):
    """ Append file name strings ending with '.csv' into an empty list """
    csv_list = [files for files in sorted(os.listdir(file_dir))
                if files.endswith(".csv")]

    return csv_list

def get_data_dict(file_dir):
    """ Returns a dictionary of dataframes """
    csv_dict = {f: pd.read_csv(file_dir+file) for f, file in
                enumerate(get_filename_list(file_dir))}

    return csv_dict

def get_dummy(data, column_list, drop_first=False):
    """ Get dummy variables of data2 and return concatenated
        data with data 1 """
    new_data = data.copy()
    for dummy in column_list:
        dummies = pd.get_dummies(data[dummy], prefix=dummy,
                                 drop_first=drop_first)
        new_data = pd.concat([new_data, dummies], axis=1)
        new_data = new_data.drop(dummy, axis=1)

    return new_data


def get_numerical(file_dir, x_col, y_cols):
    """ Takes in file path, target columns and
        extra feature columns. Returns transformed target dataset.  
    """
    data_dict = get_data_dict(file_dir)
    weekly_dict = {}
    for d in range(len(data_dict)):
        data = data_dict[d]
        data_x = data[[col for col in data.columns if x_col in col]]
        data_x = data_x.applymap(lambda x: 
                        len(','.join(str(x).lower().split(';')).split(','))
                            if len(str(x)) > 3 else 0)
        data_x['total_open'] = np.sum(data_x, axis=1)
        weekly_dict[d] = pd.concat([data_x[['total_open']],
                                    data[y_cols]],axis=1)
    return weekly_dict     


def convert_datestr(date_dict, data, date_col):
    new_data = data.copy()

    new_data[date_col] = new_data[date_col].apply(
        lambda x: ' '.join([x[x.find(date):-1]
                            for date in date_dict.keys()
                            if date in x]).split()[:2])

    new_data[date_col] = new_data[date_col].map(
        lambda x: '-'.join(list(date_dict[x[0]])+list(x[1:]))+'-2020'
        if x != [] else 'No Date')
    
    return new_data


def scale_columns(data, columns, get_dict=True, round_to=4):
    """ Returns new data with column values converted to scaled values
        with mean of 0 and standard deviation of 1. If get_dict==False,
        return the new data only.

        Params
        ------
        data(DataFrame)
        columns(str)
        get_dict(bool)
        round_to(int): decimal value to round up to
    """
    scaled_dict = {}
    data_ = data.copy()
    
    for each in columns:
        mean, std = data_[each].mean(), data_[each].std()
        scaled_dict[each] = [mean, std]
        data_.loc[:, each] = round((data_[each] - mean)/std, round_to)

    if get_dict == False:
        return data_

    else:
        return data_, scaled_dict


def get_sequences(data):
    pass









