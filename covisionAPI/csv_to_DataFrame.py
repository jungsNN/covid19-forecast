#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 21:11:05 2020

@author: jungle
"""


import pandas as pd
import numpy as np

csse_csv = pd.read_csv("csv_files/covid_328.csv")
csse = csse_csv[csse_csv['Country/Region'] == 'US']
csse = csse.values[0][1:].astype('int64')


#add on new totals from march 30th to present
addnew = [20297, 24742, 26473, 29874, 32284, 34196]

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


dailytotal = dailytotal[1:]
dates = pd.date_range(start='2020-1-23', end='2020-4-4')

#first create dictionary objects to be called in other filepaths
#fbProphet requires columns ds (Date) and y (value)
dataDict = {
            'dailytotal': dailytotal,
            'y': y,
            'ds': dates
            }


#------------------------------------end here

