import time
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from mpl_toolkits import mplot3d
from yahoo_fin import options

style.use("ggplot")

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



# print(chain["puts"].head())
