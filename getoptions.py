import time
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from mpl_toolkits import mplot3d
from yahoo_fin import options
from yahoo_fin import stock_info

style.use("ggplot")
stock = input("Input Stock Ticker:  ")
ticker = stock_info.tickers_sp500()
index = 0
for t in ticker:
    print(index)
    stock = t
    datess = options.get_expiration_dates(stock)
    # datess = ['03/06/2020','03/13/2020','03/20/2020','03/27/2020','04/03/2020','04/17/2020','05/15/2020','06/19/2020','07/17/2020','08/21/2020','09/18/2020','10/16/2020','01/15/2021','03/19/2021','06/18/2021','09/17/2021','01/21/2022','03/18/2022','06/17/2022']
    days = []
    days1 = []
    volatility = []
    volatility1 = []
    calls = pd.DataFrame()
    puts = pd.DataFrame()
    for i in range(len(datess)):
        d = datess[i]
        try:
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
            calls = pd.concat([calls,data])
            puts = pd.concat([puts,data1])
            # data = data[data['Maturity'] != 0]
        except Exception:
            print("error")
    calls["Maturity"] = days
    puts["Maturity"] = days1
    calls['Implied Volatility'] = volatility
    puts['Implied Volatility'] = volatility1
    calls = calls[calls['Implied Volatility'] != 0]
    puts = puts[puts['Implied Volatility'] != 0]
    # for x in calls['Maturity']:
    #     x = int(x)

    # for x in calls['Strike']:
    #     x  = int(x)

    calls.to_csv('data/' + stock + 'calls.csv')
    puts.to_csv('data/' + stock + 'puts.csv')
    index += 1
# # # print(chain["puts"].head())
