
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

"""
ATTAIN US' DAILY TOTAL CONFIRMED CASES, GENERATE PERCENT CHANGE RATES FOR EACH DAY AS WELL AS SEVEN DAY
FORECAST. FOR SIMPLICITY, ONLY THE SEVENTH DAY PREDICTION WILL BE ACCESSED. CURRENT FILE REDUCES TO 
SINGLE DICTIONARY OF 4 ITEMS: 
1. TODAY'S TOTAL
2. RATE CHANGE SINCE PREVIOUS DAY
3. PREDICTED RATE CHANGE FROM TODAY'S NUMBER AND EXACTLY ONE WEEK FROM TODAY
4. LAST DATE/TIME OF UPDATE

THE RESULTED DICTIONARY CAN BE USED AS REQUESTS PARAMETER AHD BE CONVERTED INTO JSON

*rough draft
"""

csse_csv = pd.read_csv("covid_328.csv")
csse_csv.columns.values
csse = pd.DataFrame(csse_csv).groupby('Country/Region').sum()
csseindex = pd.DatetimeIndex(csse.columns.values)
csse_ = pd.DataFrame(csse, index=csse.index, columns=csseindex)
top = csse.sort_values(by='3/29/20')


#top 30 countries (based on March 29th) and their population (in mil) were collected, in initial attempt
#to compare global confirmed vs US confirmed (in scale of each country's proportion to global pop).
#DISREGARD. ONLY WAS USED FOR SELECTING US DATA

get_top = top[-30:]

popbymil_d = {'Mexico':128.93,
           'Norway':5.42,
           'Brazil':212.56,
           'Israel':8.66,
           'Portugal':10.20,
           'Canada':37.74,
           'Austria':9.01,
           'S.Korea':51.27,
           'Turkey':84.34,
           'Netherlands':17.14,
           'Belgium':11.59,
           'Switzerland':8.66,
           'UK':67.89,
           'Iran':83.99,
           'France':65.27,
           'Germany':83.78,
           'China':1439.32,
           'Spain':46.76,
           'Italy':60.46,
           'US':331.00
           }

topdf = csse_[csse_.index.isin(popbymil_d.keys())]

us1 = topdf[topdf.index == 'US']

from datetime import datetime
from datetime import date

today = datetime.today()
today.isoformat()
dateindex = pd.to_datetime(csse.columns)

us = us1.values[0]

#original csv file only contained up to March 29th. Need to append numbers posted up to current date
addnew = [20297, 24742, 26473, 29874, 32284]
usdailynp = us.copy()
uspctchange = pd.Series(usdailynp).pct_change()
uspctchange = uspctchange.apply(lambda x: x*100)
usdailyvals = uspctchange.values


for n in range(len(addnew)):
    usdailynp = np.append(usdailynp, usdailynp[-1]+addnew[n])
updated_dates = pd.date_range(start='2020-1-23', end='2020-4-3')


# Create a DataFrame. fbProphet requires columns ds (Date) and y (value)
columns = ['ds']
usdaily = pd.DataFrame(updated_dates, columns=columns)
usdaily['y'] = usdailyvals[1:]
usdaily['Total'] = usdailynp[1:]
usdaily_ = usdaily.copy()


#-----------------------------
def inc_or_dec(number, comparable):
    phrase = ''
    
    if number > comparable:
        phrase += 'increase'
    elif number < comparable:
        phrase += 'decrease'
    elif number == comparable:
        phrase += 'change in rate'
    else:
        print('error')
    return phrase




#plt.plot(usdaily_['ds'], usdaily_['y'], 'b-')
#plt.plot(usdailyforecast['ds'], usdailyforecast['trend'], 'r-')
#plt.plot(usdailyforecast['ds'], usdailyforecast['yhat'], 'g-')
#plt.plot(usdailyforecast['Date'], usdailyforecast['us_yhat'], 'r-')
#-----------------------------

#HERE, I WOULD REQUEST&APPEND THE DATE&NEW TOTAL PCT CHANGE*100 TO USDAILY

import requests

r = requests.get("https://corona.lmao.ninja/states")
r.status_code
data = r.json()

todaytotal = 0
for x in data:
    for key,val in x.items():
        if key == str('todayCases'):
            todaytotal += val

#check for 'True' if today's date is in usdaily_ 'ds' column
today in usdaily_['ds'].values

if today in usdaily_['ds'].values and todaytotal != 0:

    updatedvalue = usdaily_['Total']
    updatedvalue.values[-1] += todaytotal - updatedvalue.values[-1]
    updatepct = updatedvalue.pct_change()
    updatepct = updatepct.apply(lambda v: v*100)
    updatedpct = updatepct.values[-1]
    usdaily_['y'].values[-1] = updatedpct
    usdaily_['Total'].values[-1] = updatedvalue.values[-1]
    

elif today not in usdaily_['ds'].values: 
    
    updatedtotal = usdaily_['Total'].values[-1]+todaytotal
    ustotalvals = pd.Series(np.append(usdaily_['Total'].values, updatedtotal))
    updatepct_ = ustotalvals.pct_change()
    updatepct_ = updatepct_.apply(lambda v: v*100)
    updatedpct_ = updatepct_.values[-1]
    usdaily_ = usdaily_.append({'y':updatedpct_,'ds':date.today(), 'Total':updatedtotal}, ignore_index=True)
    
    
import fbprophet


#higher changepoints & scale sought for better fit after reviewing plots
usdailyprophet = fbprophet.Prophet(changepoint_prior_scale=0.7, n_changepoints=20)
usdailyprophet.fit(usdaily_)



#make future dataframe for next seven days
usdailyforecast = usdailyprophet.make_future_dataframe(periods=1*7, freq='D')
usdailyforecast = usdailyprophet.predict(usdailyforecast)

#forecast rate change from last updated total and predicted seventh date total
forecastrate =  pd.Series([usdailyforecast['trend'].values[-8],usdailyforecast['trend'].values[-1]]).pct_change()
forecastrate = forecastrate[1]*100

#create a dictionary that will be used for requests
to_json = {}

yval = usdaily_['y'].values[-1]
newcases = "New cases today in US: {}".format(todaytotal)
changerate = '{:.2f}% {} from yesterday'.format(yval, inc_or_dec(usdaily_['Total'].values[-1], usdaily_['Total'].values[-2]))
nextweek = 'Forecast on week from today: {:.2f}'.format(forecastrate, inc_or_dec(forecastrate, 0))
timenow = 'Updated at {}'.format(datetime.today().isoformat())

to_json['newcases'] = str(newcases)
to_json['changerate'] = str(changerate)
to_json['nextweek'] = str(nextweek)
to_json['timenow'] = str(timenow)

