#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 21:11:05 2020
~
@author: jungle
"""


import pandas as pd
import numpy as np

csse_csv = pd.read_csv("covid_328.csv")
csse = csse_csv[csse_csv['Country/Region'] == 'US']
csse = csse.values[0][1:].astype('int64')


#add on new totals from march 30th to present
addnew = [20297, 24742, 26473, 29874, 32284, 34196, 25316]

dailytotal = csse.copy()
for n in range(len(addnew)):
    dailytotal = np.append(dailytotal, dailytotal[-1]+addnew[n])


#get percent rate changes of each day to its following day
len(dailytotal)


y = []
breaker = 0

while breaker < len(dailytotal[:-1]):
    for e in range(len(dailytotal[:-1])):
        y_ = pd.Series([dailytotal[e], dailytotal[e+1]]).pct_change()
        y.append(y_[1] * 100)
        breaker += 1   
breaker

dates = pd.date_range(start='1/23/2020', end='4/5/2020')
#format dates into string for the dictionary
datesUpdate = pd.Series(dataDict['ds'])
dateformat = dates.values
dateformat
datestr = np.datetime_as_string(dateformat)
datestr
datestr = [s[:10] for s in datestr]
datestr


dailytotal = dailytotal[1:]
#dates = pd.date_range(start='2020-1-23', end='2020-4-5')

#first create dictionary objects to be called in other filepaths
#fbProphet requires columns ds (Date) and y (value)
dataDict = {
            'dailytotal': dailytotal,
            'y': y,
            'ds': datestr
            }


#------------------------------------

dataDF = pd.DataFrame(dataDict)


import requests

url = "https://covid-19-data.p.rapidapi.com/country"

querystring = {"format":"undefined","name":"usa"}

headers = {
    'x-rapidapi-host': "covid-19-data.p.rapidapi.com",
    'x-rapidapi-key': "cd23e16bb3msh2894958d714157cp1732d3jsn36305d103c76"
    }

response = requests.request("GET", url, headers=headers, params=querystring)

data = response.json()
data = data[0]
data
confirmed = 0


for key,val in data.items():
    if key == str('confirmed'):
        confirmed += val
confirmed

import fbprophet
from datetime import datetime
from datetime import date

today = date.today()
today = today.isoformat()


if today not in dataDict['ds']:
    dataDict['ds'].append(today)
    updatepct = pd.Series([dataDict['dailytotal'][-1], confirmed]).pct_change()
    updatepct = updatepct[1]*100
    dataDict['y'] = np.append(dataDict['y'], updatepct)
    dataDict['dailytotal'] = np.append(dataDict['dailytotal'], confirmed)
else:  
    dataDict['dailytotal'][-1] = confirmed
    updatepct = pd.Series([dataDict['dailytotal'][-2], dataDict['dailytotal'][-1]]).pct_change()
    updatepct = updatepct[1]*100
    dataDict['y'][-1] = updatepct


dataDF = pd.DataFrame(dataDict)


dailyprophet = fbprophet.Prophet(changepoint_prior_scale=0.7, n_changepoints=20)
dailyprophet.fit(dataDF)

#usdailyprophet.changepoints[:10]
# Make a future dataframe for 2 years
dailyforecast = dailyprophet.make_future_dataframe(periods=1*7, freq='D')
dailyforecast = dailyprophet.predict(dailyforecast)

changerate =  pd.Series([dailyforecast['trend'].values[-8],dailyforecast['trend'].values[-1]]).pct_change()
changerate = changerate[1]*100
changerate
#THEN HERE, YOU'D ADD THE LAST VALUES OF USDAILY&USDAILYFORECAST TO JSON
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



to_json = {}
yval = dataDF['y'].values[-1]


newcases = "New cases today in US: {}".format(confirmed-dataDF['dailytotal'].values[-2])
changeRate = '{:.2f}% {} from yesterday'.format(yval, inc_or_dec(dataDF['dailytotal'].values[-1], dataDF['dailytotal'].values[-2]))
nextweek = 'Forecast on week from today: {:.2f}'.format(changerate, inc_or_dec(changerate, 0))
timenow = 'Updated at {}'.format(datetime.today().isoformat())
to_json['newcases'] = str(newcases)
to_json['changerate'] = str(changeRate)
to_json['nextweek'] = str(nextweek)
to_json['timenow'] = str(timenow)

print(to_json)
        

        
        
        
        
        
        
        
    
