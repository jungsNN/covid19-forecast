import pandas as pd
import csv
import numpy as np
import boto3
import datetime
import os
import torch

import forecastnet
import config

path = os.getcwd()
SERVE_DIR = os.path.join(path, 'mysite/serve/')
# AWS S3
PUBLIC_BUCKET_NAME = 'sagemaker-us-east-1-476204846640'
PUBLIC_PREFIX = 'nyt-states-reopen-status-covid-19/dataset/nyt-states-reopen-status-covid-19.csv'
# New York Times Daily Github Raw Data
RAW_URL = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'
CSV = [f for f in os.listdir('mysite/train/') if f.endswith('.csv') and         f.startswith('forecast_')][0]
LAST_DATA = pd.read_csv('mysite/train/'+CSV)
# Status (target) mask
STATUS_DICT = {1: ['reopened', 'forward', 'all'],
               2: ['reopening', 'pausing', 'regional', 'soon'],
               3: ['reversing', 'shutdown-restricted']}
''' -------- MODEL CONFIG -------- '''
IN_FEATURES = 58
OUT_FEATURES = 6
N_STEPS = 7
N_LAYERS = 2
HID_DIM = 1024

DEVICE = torch.device('cpu')

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
    x = data.query("state in @LAST_DATA.state").sort_values(by=['state', 'date']).set_index('date')
    fips_dummy = pd.get_dummies(x['fips'], prefix='fips')
    scaled = normalize(x[['cases', 'deaths']])
    x = pd.concat((x[['state']], fips_dummy, scaled), axis=1)
    return x


def preprocess_y(data):
    y = data.query("state in @LAST_DATA.state").sort_values(by='state')
    y = y.reset_index(drop=True)
    embed = np.zeros((len(y), 3), dtype=int)
    status = mask_status(y, 'status', STATUS_DICT)
    for i in range(len(status)):
        embed[i][status[i] - 1] = 1

    extracted = extract_cols(y, 'opened')
    y['total_open'] = np.sum(count_string_values(extracted), axis=1)
    y[['status_1', 'status_2', 'status_3']] = pd.DataFrame(embed)
    y = pd.concat([y[['state']],
                    normalize(y[['population']]),
                    normalize(y[['total_open']]),
                    y[['status_1', 'status_2', 'status_3']]], axis=1)
    return y


def get_inputs(date_from, reopen_data):
    x = pd.read_csv(RAW_URL, error_bad_lines=False)
    x = x.loc[x.date >= date_from]
    if datetime.date.today() not in x.date:

        missing_date_rolling = (x.set_index(pd.to_datetime(x['date']))
                                                            .groupby('state')[['cases', 'deaths']]
                                                            .resample('6D')
                                                            .mean()
                                                            .reset_index()
                                                            .drop('date', axis=1))
        today = [datetime.date.today().strftime("%Y-%m-%d")] * len(missing_date_rolling)
        missing_date_rolling = pd.concat([pd.DataFrame(today, columns=['date']), missing_date_rolling], axis=1)
        x.append(missing_date_rolling)

    input_data = preprocess_x(x).reset_index().merge(preprocess_y(reopen_data),
                                                                how='left', on='state')
    daily = []
    for date in sorted(input_data.date.unique()):
        daily.append(input_data.loc[input_data.date == date].drop(['date', 'state'],
                                                                    axis=1).reset_index(drop=True))

    return pd.concat(daily, axis=1).values


def get_raw_forecast(inputs, model): # PICK BACK UP FROM HERE: NEED TO MODIFY THE INPUT DATASETS
    """
    returns arrays
    """
    model.float()
    h = model.init_hidden(len(inputs))
    output, _ = model(torch.from_numpy(inputs).float().to(DEVICE), h)
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

    return serialized.reset_index(drop=True)


def save_forecast(data):
    fips_state_cols = LAST_DATA[['state', 'fips']].sort_values(by='state').reset_index(drop=True)
    rows = pd.concat([fips_state_cols, data], axis=1).values
    filename = 'forecast_{}.csv'.format(datetime.date.today().strftime("%Y%m%d"))
    with open(os.path.join(SERVE_DIR, filename), 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['state', 'fips', 'cases', 'deaths', 'total_open', 'status'])
        csvwriter.writerows(rows)


def save_today_data(y):
    # rearc NYT reopen data
    today_y = y.query("state in @LAST_DATA.state")
    today_status = today_y.iloc[:, -3:]
    today_opened = extract_cols(today_y, 'opened')
    today_y['total_open'] = np.sum(count_string_values(today_opened), axis=1)
    today_y = today_y.sort_values(by='state')[['population', 'total_open', 'status']]

   # raw daily data
    today_x = pd.read_csv(RAW_URL, error_bad_lines=False).query("state in @LAST_DATA.state")
    most_recent_date = sorted(today_x.date.unique())[-1]
    most_recent_date
    today_x = today_x.loc[today_x.date == most_recent_date]

    # merged returns ['state', 'fips', 'cases', 'deaths', 'population', 'total_open', 'status'] actual today values
    today_data = pd.concat([today_x.iloc[:, 1:].sort_values(by='state').reset_index(drop=True),
                            today_y.reset_index(drop=True)], axis=1)

    save_path = 'raw_daily_{}.csv'.format(datetime.date.today())
    with open(os.path.join(SERVE_DIR, save_path), 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['state', 'fips', 'cases', 'deaths', 'population', 'total_open', 'status'])
        csvwriter.writerows(today_data.values)
        

def run_app():
    nyt_reopen_filepath = os.path.join(SERVE_DIR, 'nyt_reopen_{}.csv'.format( datetime.date.today().strftime("%Y%m%d")))

    s3 = boto3.resource('s3', aws_access_key_id=config.SECRET_KEY,
                        aws_secret_access_key=config.ACCESS_KEY,
                        region_name='us-east-1')
    s3.Bucket(PUBLIC_BUCKET_NAME).download_file(PUBLIC_PREFIX, nyt_reopen_filepath)

    date_from = today - datetime.timedelta(7)
    week_inputs = get_inputs(date_from.strftime("%Y-%m-%d"), pd.read_csv(nyt_reopen_filepath))
    reopen_data = pd.read_csv(nyt_reopen_filepath)

    MODEL = forecastnet.ForecastLSTM(IN_FEATURES, OUT_FEATURES, N_STEPS, N_LAYERS, HID_DIM)
    MODEL.load_state_dict(torch.load('mysite/forecast_lstm.pt', map_location=DEVICE))
    MODEL.to(DEVICE)
    MODEL.eval()

    save_forecast(serialize_forecast(get_raw_forecast(week_inputs, MODEL)))

# def test_model():
#     nyt_reopen_filepath = os.path.join(SERVE_DIR, 'nyt_reopen_{}.csv'.format( datetime.date.today().strftime("%Y%m%d")))
#     s3 = boto3.resource('s3', aws_access_key_id=config.SECRET_KEY,
#                         aws_secret_access_key=config.ACCESS_KEY,
#                         region_name='us-east-1')
#     s3.Bucket(PUBLIC_BUCKET_NAME).download_file(PUBLIC_PREFIX, nyt_reopen_filepath)
#     reopen_data = pd.read_csv(nyt_reopen_filepath)
#     inputs = get_inputs(RAW_URL, reopen_data)
#     MODEL = forecastnet.ForecastNet(INP_DIM, OUT_DIM, HID_DIMS)
#     MODEL.load_state_dict(torch.load('mysite/forecastnet.pt', map_location=DEVICE))
#     MODEL.to(torch.device('cpu'))
#     MODEL.eval()

#     save_forecast(serialize_forecast(get_raw_forecast(inputs, MODEL)))


if __name__ == "__main__":
    run_app()
