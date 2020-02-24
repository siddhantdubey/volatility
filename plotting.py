import time
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from mpl_toolkits import mplot3d
from yahoo_fin import options

style.use("ggplot")

data = pd.read_csv('calls.csv')

# threedee = plt.figure().gca(projection='3d')
# threedee.scatter(data['Maturity'], data['Strike'], data['Implied Volatility'])
# threedee.set_xlabel('Maturity')
# threedee.set_ylabel('Strike')
# threedee.set_zlabel('Volatility')

data.plot(x="Strike", y="Implied Volatility", kind="line")
plt.show()
