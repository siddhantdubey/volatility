import time
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from mpl_toolkits import mplot3d
from yahoo_fin import options

style.use("ggplot")
<<<<<<< HEAD
datess = ['03/06/2020','03/13/2020','03/20/2020','03/27/2020','04/03/2020','04/17/2020','05/15/2020','06/19/2020','07/17/2020','08/21/2020','09/18/2020','10/16/2020','01/15/2021','03/19/2021','06/18/2021','09/17/2021','01/21/2022','03/18/2022','06/17/2022']
for i in range(len(datess)):
    d = datess[i]
    chain = options.get_options_chain("TSLA", d)

    data = pd.DataFrame(chain['calls']) 

    expiry = d[6:] + '-' + d[:2] + '-' + d[3:5] #'2020-03-20'

    days = []
    for x in chain['calls']['Last Trade Date']:
        days.append(int((dt.datetime.strptime(expiry, '%Y-%m-%d') - dt.datetime.today()).days))

    data["Maturity"] = days
    data1 = pd.DataFrame(chain['puts'])
    days1 = []
    for x in chain['puts']['Last Trade Date']:
        days1.append(int((dt.datetime.strptime(expiry, '%Y-%m-%d') - dt.datetime.today()).days))

    data1["Maturity"] = days1
    for x in data['Maturity']:
        x = int(x)

    for x in data['Strike']:
        x  = int(x)
    volatility = []
    for x in data['Implied Volatility']:
        stringy = x[:-1]
        stringy = stringy.replace(',','')
        x = float(stringy)
        volatility.append(x)

    data['Implied Volatility'] = volatility
    # data = data[data['Maturity'] != 0]
    data.to_csv('calls' + d[6:] + '-' + d[:2] + '-' + d[3:5] + '.csv')
    data1.to_csv('puts' + d[6:] + '-' + d[:2] + '-' + d[3:5] + '.csv')
=======

chain = options.get_options_chain("TSLA", '03/20/2020')

data = pd.DataFrame(chain['calls']) 

expiry = '2020-03-20'

days = []
for x in chain['calls']['Last Trade Date']:
    days.append(int((dt.datetime.strptime(expiry, '%Y-%m-%d') - dt.datetime.strptime(x[:10], '%Y-%m-%d')).days))

data["Maturity"] = days
data1 = pd.DataFrame(chain['puts'])
days1 = []
for x in chain['puts']['Last Trade Date']:
    days1.append(int((dt.datetime.strptime(expiry, '%Y-%m-%d') - dt.datetime.strptime(x[:10], '%Y-%m-%d')).days))

data1["Maturity"] = days1
for x in data['Maturity']:
    x = int(x)

for x in data['Strike']:
    x  = int(x)
volatility = []
for x in data['Implied Volatility']:
    x = float(x[:-1])
    volatility.append(x)

data['Implied Volatility'] = volatility
# data = data[data['Maturity'] != 0]
data.to_csv('calls.csv')
data1.to_csv('puts.csv')
>>>>>>> 35d3a6116125426a249dd72704b9e894264ed77e



# print(chain["puts"].head())
