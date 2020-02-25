import time
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from mpl_toolkits import mplot3d
from yahoo_fin import options

style.use("ggplot")
stock = "TSLA"
datess = options.get_expiration_dates(stock)
# datess = ['03/06/2020','03/13/2020','03/20/2020','03/27/2020','04/03/2020','04/17/2020','05/15/2020','06/19/2020','07/17/2020','08/21/2020','09/18/2020','10/16/2020','01/15/2021','03/19/2021','06/18/2021','09/17/2021','01/21/2022','03/18/2022','06/17/2022']
days = []
days1 = []
volatility = []
volatility1 = []
tdata = pd.DataFrame()
tdata1 = pd.DataFrame()
for i in range(len(datess)):
    d = datess[i]
    chain = options.get_options_chain(stock, d)

    data = pd.DataFrame(chain['calls']) 
    
    for x in chain['calls']['Last Trade Date']:
        days.append(int((dt.datetime.strptime(d, '%B %d, %Y') - dt.datetime.today()).days))

    data1 = pd.DataFrame(chain['puts'])
    
    for x in chain['puts']['Last Trade Date']:
        days1.append(int((dt.datetime.strptime(d, '%B %d, %Y') - dt.datetime.today()).days))
    for x in data['Implied Volatility']:
        stringy = x[:-1]
        stringy = stringy.replace(',','')
        x = float(stringy)
        volatility.append(x)
    for x in data1['Implied Volatility']:
        stringy = x[:-1]
        stringy = stringy.replace(',','')
        x = float(stringy)
        volatility1.append(x)
    tdata = pd.concat([tdata,data])
    tdata1 = pd.concat([tdata1,data1])
    # data = data[data['Maturity'] != 0]
tdata["Maturity"] = days
tdata1["Maturity"] = days1
tdata['Implied Volatility'] = volatility
tdata1['Implied Volatility'] = volatility1
tdata = tdata[tdata['Implied Volatility'] != 0]
tdata1 = tdata1[tdata1['Implied Volatility'] != 0]
# for x in tdata['Maturity']:
#     x = int(x)

# for x in tdata['Strike']:
#     x  = int(x)

tdata.to_csv('data/' + stock + 'calls.csv')
tdata1.to_csv('data/' + stock + 'puts.csv')
# # # print(chain["puts"].head())
