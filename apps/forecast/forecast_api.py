import pandas as pd
import csv
import numpy as np
import boto3
import datetime
import os
import torch
from .static.resources.forecast.serve import forecastnet


SERVE_DIR = 'forecast/static/serve'
PUBLIC_BUCKET_NAME = 'nyt-reopen-dx-datas3bucket-1gc7v1ltd3dtf'
PUBLIC_PREFIX = 'nyt-states-reopen-status-covid-19/dataset/nyt-states-reopen-status-covid-19.csv'
RAW_URL = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'
CSV = [f for f in os.listdir(SERVE_DIR) if f.endswith('.csv')][0]
LAST_DATA = pd.read_csv(os.path.join(SERVE_DIR, CSV))
STATUS_DICT = {1: ['reopened', 'forward', 'all'],
               2: ['reopening', 'pausing', 'regional', 'soon'],
               3: ['reversing', 'shutdown-restricted']}
''' -------- MODEL -------- '''
INP_DIM = 58
OUT_DIM = 6
HID_DIMS = [256, 512]
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
state_dict_path = os.path.join(SERVE_DIR, 'forecastnet.pt')
MODEL = forecastnet.ForecastNet(INP_DIM, OUT_DIM, HID_DIMS)
MODEL.load_state_dict(torch.load(state_dict_path, map_location=torch.device('cpu')))
torch.set_grad_enabled(False)
MODEL.to(DEVICE)
MODEL.eval()


def load_csv(filepath):
    data_rows = []
    with open(filepath, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for lines in csvreader:
            data_rows.append(lines)
    data = pd.DataFrame(data_rows[1:], columns=data_rows[0])
    return data


def normalize(data):
    data_ = data.copy()
    for var in data_.columns:
        max_val = data[var].max()
        data_.loc[:, var] = data[var] / max_val
        np.save(os.path.join(SERVE_DIR, 'max_{}.npy'.format(var)), max_val)

    return data_


def mask_status(data, column, mask_dict):
    mask = 1
    masked = data[column].apply(lambda x: mask if x in mask_dict[mask]
    else (mask + 1 if x in mask_dict[mask + 1]
          else mask + 2))

    return masked


def extract_cols(data, extract_str):
    return data[[col for col in data.columns if extract_str in col]]


def count_string_values(data, exclude_parser=';', val_len=3):
    exp = exclude_parser
    return data.applymap(lambda x: len(','.join(str(x).lower().split(exp))
                                       .split(',')) if len(str(x)) > val_len else 0)


def preprocess_x(data):
    latest_date = data.iloc[-1, 0]
    x = data.loc[data.date == latest_date]
    x = x.query("fips in @LAST_DATA.fips").sort_values(by='state')
    fips_dummy = pd.get_dummies(x['fips'], prefix='fips').values
    scaled = normalize(x[['cases', 'deaths']]).values
    x = np.concatenate((fips_dummy, scaled), axis=1)
    return x


def preprocess_y(data):
    y = data.query("state in @LAST_DATA.state").sort_values(by='state').drop('state', axis=1)
    y = y.reset_index(drop=True)
    embed = np.zeros((len(y), 3), dtype=int)
    status = mask_status(y, 'status', STATUS_DICT)
    for i in range(len(status)):
        idx = int()
        embed[i][status[i] - 1] = 1

    extracted = extract_cols(y, 'opened')
    extracted = np.sum(count_string_values(extracted), axis=1)
    y = np.concatenate([normalize(y[['population']]).values,
                        normalize(pd.DataFrame(extracted)).values,
                        embed], axis=1)
    return y


def get_inputs(x_raw, reopen_data):
    x = pd.read_csv(x_raw, error_bad_lines=False)
    input_data = np.concatenate([preprocess_x(x), preprocess_y(reopen_data)], axis=1)

    return input_data


def get_raw_forecast(inputs, model):
    """
    returns arrays
    """
    output = model(torch.from_numpy(inputs).float().to(DEVICE))
    output = output.detach().numpy()
    return output


def serialize_dummy(arr):
    out_list = []
    for row in arr:
        out_list.extend([int(i + 1) for i in range(len(row)) if row[i] == row.max(0)])
    return np.array(out_list)


def serialize_forecast(raw_outputs):
    """
    max values in the order of ['cases', 'deaths', 'total_open']
    """

    max_vals = [np.load(os.path.join(SERVE_DIR, 'max_cases.npy')),
                np.load(os.path.join(SERVE_DIR, 'max_deaths.npy')),
                np.load(os.path.join(SERVE_DIR, 'max_total_open.npy'))]
    serialized = raw_outputs[:, :-3]
    status = serialize_dummy(raw_outputs[:, -3:])
    for mx in range(len(max_vals)):
        inversed = serialized[:, mx] * max_vals[mx]
        serialized[:, mx] = [int(round(x)) for x in inversed]
    serialized = pd.concat([pd.DataFrame(serialized), pd.DataFrame(status)], axis=1)
    for npfile in os.listdir(SERVE_DIR):
        if npfile.endswith('.npy'):
            os.remove(os.path.join(SERVE_DIR, npfile))
    return serialized.reset_index(drop=True)


def save_forecast(data):
    fips_state_cols = LAST_DATA[['state', 'fips']].sort_values(by='state').reset_index(drop=True)
    rows = pd.concat([fips_state_cols, data], axis=1).values
    filename = 'forecast_{}.csv'.format(datetime.date.today().strftime("%Y%m%d"))
    with open(os.path.join(SERVE_DIR, filename), 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['state', 'fips', 'cases', 'deaths', 'total_open', 'status'])
        csvwriter.writerows(rows)


def run_app():
    nyt_reopen_filename = 'nyt_reopen_{}'.format( datetime.date.today().strftime("%Y%m%d"))
    s3 = boto3.resource('s3')
    s3.Bucket(PUBLIC_BUCKET_NAME).download_file(PUBLIC_PREFIX, nyt_reopen_filename)
    reopen_data = pd.read_csv(os.path.join(SERVE_DIR, nyt_reopen_filename))
    inputs = get_inputs(RAW_URL, reopen_data)
    save_forecast(serialize_forecast(get_raw_forecast(inputs, MODEL)))
    os.remove(os.path.join(SERVE_DIR, nyt_reopen_filename))

