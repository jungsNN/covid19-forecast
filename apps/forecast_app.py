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
    except:
        forecast_api.run_app()
        data = pd.read_csv(os.path.join(app.config['SERVE_DIR'], forecast_filepath))
    data.state = data.state.apply(lambda x: '_'.join(str(x).split()) if ' ' in str(x) else str(x))
    data = pd.DataFrame(data.loc[data.fips == fips, :]).to_string().split()[-6:]


    return data

 @app.route('/', methods=['POST', 'GET'])
 def forecast():
     next_week = datetime.date.today() + datetime.timedelta(7)
     if request.method=='POST':
         today_date = datetime.date.today().strftime("%Y%m%d")
         vals = [str(x) for x in range(57)]
         result = ''
         state = ''
         fips = ''
         cases = ''
         deaths = ''
         total_open = ''
         status = ''
         forecast_filepath = 'forecast_{}.csv'.format(today_date)
         q = request.form.get('fips')
         if q in vals:
             result += q
             outputs = query_state(result, forecast_filepath)
             state += outputs[0]
             fips += outputs[1]
             cases += outputs[2]
             deaths += outputs[3]
             total_open += outputs[4]
             status += outputs[5]

         return render_template('forecast.html', next_week=next_week, state=state, fips=fips, cases=cases,
                                 deaths=deaths, total_open=total_open, status=status)
     else:
         return render_template('forecast.html', next_week=next_week)
#
#@app.route('/')
#def maintenance():
#    # try:
#    #     test_result = pd.read_csv('mysite/serve/forecast_{}.csv'.format(datetime.date.today().strftime("%Y%m%d"))).to_string()
#    # except FileNotFoundError:
#    #     forecast_api.test_model()
#    #     test_result = pd.read_csv('mysite/serve/forecast_{}.csv'.format(datetime.date.today().strftime("%Y%m%d"))).to_string()
#    return render_template("test_page.html")
