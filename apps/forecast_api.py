import pandas as pd
import csv
import numpy as np
import boto3
import datetime
import os
import torch

import forecastnet
from data_utils import *
import config
# NOTE: THESE METHODS ARE HIGHLY PERTAINED TO NEW YORK TIMES DATASETS

def preprocess_x(data):
    """
    Returns a preprocessed 'x' data, which is retrieved from NYT Covid-19 Github repository.
    It includes important information, such as 'cases' and 'deaths' for each day.
    """
    x = data.query("state in @LAST_DATA.state").sort_values(by=['state', 'date']).set_index('date')
    fips_dummy = pd.get_dummies(x['fips'], prefix='fips')
    scaled = normalize(x[['cases', 'deaths']])
    x = pd.concat((x[['state']], fips_dummy, scaled), axis=1)
    return x


def preprocess_reopen(data):
    """
    Returns a preprocessed NYT Reopen/Closed dataset from AWS dataexchange.
    The data is referred to as 'y', because of the main target 'status' column is
    included in the data.
    """
    y = data.query("state in @LAST_DATA.state").sort_values(by='state')
    y = y.reset_index(drop=True)
    embed = np.zeros((len(y), 3), dtype=int)
    status = mask_status(y, 'status', STATUS_DICT)
    for i in range(len(status)):
        embed[i][status[i] - 1] = 1

    extracted = extract_cols(y, 'opened')
    y.loc[:, 'total_open'] = np.sum(count_string_values(extracted), axis=1)
    y[['status_1', 'status_2', 'status_3']] = pd.DataFrame(embed)
    y = pd.concat([y[['state']],
                    normalize(y[['population']]),
                    normalize(y[['total_open']]),
                    y[['status_1', 'status_2', 'status_3']]], axis=1)
    return y


def get_inputs(reopen_data):
    """
    Takes raw x and y data and prepares them as input data to the LSTM model.
    """
    x = pd.read_csv(RAW_URL, error_bad_lines=False)
    past_week = sorted(x.date.unique())[-7:]
    x = x.query("date in @past_week")

    input_data = preprocess_x(x).reset_index().merge(preprocess_reopen(reopen_data), how='left', on='state')
    daily = []
    for date in sorted(input_data.date.unique()):
        daily.append(input_data.loc[input_data.date == date].drop(['date', 'state'], axis=1).reset_index(drop=True))

    return pd.concat(daily, axis=1).values


def get_raw_forecast(inputs, model):
    """
    Takes in preprocessed input data and returns raw output from the LSTM model
    """
    model.float()
    h = model.init_hidden()
    output, _ = model(torch.from_numpy(inputs).float().detach().cpu(), h)

    return np.numpy(output)


def serialize_forecast(raw_outputs):
    """
    Returns a translated data from raw, by inversing scaled values and dummy values.
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
    """
    Specific for the NYT dataset, the function saves the output from the LSTM model
    as a csv file for retrieval.
    """
    fips_state_cols = LAST_DATA[['state', 'fips']].sort_values(by='state').reset_index(drop=True)
    rows = pd.concat([fips_state_cols, data], axis=1).values
    filename = 'forecast_{}.csv'.format(datetime.date.today().strftime("%Y%m%d"))
    with open(os.path.join(SERVE_DIR, filename), 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['state', 'fips', 'cases', 'deaths', 'total_open', 'status'])
        csvwriter.writerows(rows)

def save_today_data(x, y):
    """
    Saves actual data retrieved from sources.
    """
    # rearc NYT reopen data
    today_y = y.query("state in @LAST_DATA.state")
    today_status = today_y.iloc[:, -3:]
    today_opened = extract_cols(today_y, 'opened')
    today_y['total_open'] = np.sum(count_string_values(today_opened), axis=1)
    today_y = today_y.sort_values(by='state')[['total_open', 'status']]

   # raw daily data
    today_x = pd.read_csv(x, error_bad_lines=False).query("state in @LAST_DATA.state")
    most_recent_date = sorted(today_x.date.unique())[-1]
    most_recent_date
    today_x = today_x.loc[today_x.date == most_recent_date]

    # merged returns ['state', 'fips', 'cases', 'deaths','total_open', 'status'] actual today values
    today_data = pd.concat([today_x.iloc[:, 1:].sort_values(by='state').reset_index(drop=True),
                            today_y.reset_index(drop=True)], axis=1)

    save_path = 'raw_daily_{}.csv'.format(datetime.date.today())
    with open(os.path.join(SERVE_DIR, save_path), 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['state', 'fips', 'cases', 'deaths', 'total_open', 'status'])
        csvwriter.writerows(today_data.values)

def run_app():
    """
    Processes all steps to converting and merging datasets, and runs it through the LSTM model.
    Serialized outputs and real data are saved.
    """
    # -------- PREPROCESSING -------- #
    nyt_reopen_filepath = os.path.join(SERVE_DIR,
    'nyt_reopen_{}.csv'
    .format( datetime.date.today().strftime("%Y%m%d")))
    s3 = boto3.resource('s3', aws_access_key_id=config.SECRET_KEY,
                        aws_secret_access_key=config.ACCESS_KEY,
                        region_name=config.REGION_NAME)
    s3.Bucket(config.BUCKET_NAME).download_file(config.PREFIX, nyt_reopen_filepath)
    week_inputs = get_inputs(pd.read_csv(nyt_reopen_filepath))
    reopen_data = pd.read_csv(nyt_reopen_filepath)
    
    save_today_data(RAW_URL, reopen_data)  # saving today's actual data
    # -------- MODEL OUTPUT -------- #
    device = torch.device('cpu')
    model = forecastnet.ForecastLSTM(IN_FEATURES, OUT_FEATURES, N_LAYERS, HID_DIM, BATCH_SIZE)
    model.load_state_dict(torch.load(MODEL_PT, map_location=device))
    model.to(device)
    model.eval()

    save_forecast(serialize_forecast(get_raw_forecast(week_inputs, model)))


################################ PARAMS ###############
path = os.getcwd()
SERVE_DIR = os.path.join(path, 'apps/serve/')
RAW_URL = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'
CSV = [f for f in os.listdir('apps/train/') if f.endswith('.csv') and f.startswith('forecast_')][0]
LAST_DATA = pd.read_csv('apps/train/'+CSV)
STATUS_DICT = {1: ['reopened', 'forward', 'all'],
               2: ['reopening', 'pausing', 'regional', 'soon'],
               3: ['reversing', 'shutdown-restricted']}
               
IN_FEATURES = 58
OUT_FEATURES = 6
N_LAYERS = 2
HID_DIM = 1024
BATCH_SIZE = 51  # number of states
MODEL_PT = os.path.join(path, 'apps/forecast_lstm.pt')
########################################################
