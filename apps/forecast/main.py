import datetime
import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd
from . import forecast_api

''' -------------------- APP CONFIG --------------------------- '''
app = Flask(__name__)
app.secret_key = "SECRET_KEY"
path = os.getcwd()
DATA_DIR = os.path.join(path, 'forecast/static')
app.config['DATA_DIR'] = DATA_DIR

''' -------------------- APP VIEWS --------------------------- '''
def query_state(fips, forecast_filepath):
    fips = int(float(fips))
    try:
        data = pd.read_csv(os.path.join(app.config['DATA_DIR'], forecast_filepath))
    except FileNotFoundError:
        forecast_api.run_app()
        data = pd.read_csv(os.path.join(app.config['DATA_DIR'], forecast_filepath))
    data.state = data.state.apply(lambda x: '_'.join(str(x).split()) if ' ' in str(x) else str(x))
    data = pd.DataFrame(data.loc[data.fips == fips, :]).to_string().split()[-6:]

    return data

@app.route('/', methods=['POST', 'GET'])
def forecast():
    next_week = datetime.date.today() + datetime.timedelta(7)
    next_week = next_week.strftime("%Y-%m-%d")
    if request.method == 'POST':
        today_date = datetime.date.today().strftime("%Y%m%d")
        forecast_filepath = 'serve/forecast_{}.csv'.format(today_date)
        vals = [str(x) for x in range(57)]
        result = ''
        q = request.form.get('fips')
        state = ''
        fips = ''
        cases = ''
        deaths = ''
        total_open = ''
        status = ''
        if q in vals:
            result += q
            outputs = query_state(result, forecast_filepath)
            state += outputs[0]
            fips += outputs[1]
            cases += outputs[2]
            deaths += outputs[3]
            total_open += outputs[4]
            status += outputs[-1]
        return render_template("forecast.html", week=next_week, state=state, fips=fips, cases=cases,
                                deaths=deaths, total_open=total_open, status=status)
    else:
        return render_template("forecast.html", week=next_week)

