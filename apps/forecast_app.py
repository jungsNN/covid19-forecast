#!python3.8 /home/jungsnn1029/.virtualenvs/myvirtualenv/bin/python
import datetime
from flask import Flask, render_template, request
import pandas as pd
import os


import forecast_api

app = Flask(__name__)
app.config["DEBUG"] = True
path = os.getcwd()

''' ------- Forecast App Config ------- '''
SERVE_DIR = os.path.join(path,'mysite/serve/')
app.config['SERVE_DIR'] = SERVE_DIR

''' ------- Forecast App ------- '''
def query_state(fips, forecast_filepath):
    fips = int(float(fips))
    try:
        data = pd.read_csv(os.path.join(app.config['SERVE_DIR'], forecast_filepath))
        today_raw = pd.read_csv(os.path.join(app.config['SERVE_DIR'], 'raw_daily_{}.csv'.format(datetime.date.today())))
    except FileNotFoundError:
        forecast_api.run_app()
        data = pd.read_csv(os.path.join(app.config['SERVE_DIR'], forecast_filepath))
        today_raw = pd.read_csv(os.path.join(app.config['SERVE_DIR'], 'raw_daily_{}.csv'.format(datetime.date.today())))
    # Converting data into string elements
    data.state = data.state.apply(lambda x: '_'.join(str(x).split()) if ' ' in str(x) else str(x))
    data = pd.DataFrame(data.loc[data.fips == fips, :]).to_string().split()[-6:]
    today_raw.state = today_raw.state.apply(lambda x: '_'.join(str(x).split()) if ' ' in str(x) else str(x))
    today_raw = pd.DataFrame(today_raw.loc[today_raw.fips == fips, :]).to_string().split()[7:]

    return data, today_raw

@app.route('/', methods=['POST', 'GET'])
def forecast():
    next_week = datetime.date.today() + datetime.timedelta(7)
    if request.method=='POST':
        today_date = datetime.date.today().strftime("%Y%m%d")
        vals = [str(x) for x in range(57)]
        result = ''
        state = []
        fips = []
        cases = []
        deaths = []
        total_open = []
        status = []

        forecast_filepath = 'forecast_{}.csv'.format(today_date)
        q = request.form.get('fips')
        if q in vals:
            result += q
            outputs, today_state = query_state(result, forecast_filepath)
            state.extend([outputs[0], today_state[0]])
            fips.extend([outputs[1], today_state[1]])
            cases.extend([outputs[2], today_state[2]])
            deaths.extend([outputs[3], today_state[3]])
            total_open.extend([outputs[4], today_state[4]])
            status.extend([outputs[5], today_state[5]])
        return render_template('forecast.html',
                                next_week=next_week,
                                state=state[0],
                                fips=fips[0],
                                cases=cases[0],
                                deaths=deaths[0],
                                total_open=total_open[0],
                                status=status[0],
                                state_real=state[1],
                                fips_real=fips[1],
                                cases_real=cases[1],
                                deaths_real=deaths[1],
                                total_open_real=total_open[1],
                                status_real=status[1])
    else:
        return render_template('forecast.html',next_week=next_week)

# @app.route('/')
# def maintenance():
#     # try:
#     #     test_result = pd.read_csv('mysite/serve/forecast_{}.csv'.format(datetime.date.today().strftime("%Y%m%d"))).to_string()
#     # except FileNotFoundError:
#     #     forecast_api.test_model()
#     #     test_result = pd.read_csv('mysite/serve/forecast_{}.csv'.format(datetime.date.today().strftime("%Y%m%d"))).to_string()
#     return render_template("test_page.html")
