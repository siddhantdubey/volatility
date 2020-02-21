import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from yahoo_fin import options

style.use("ggplot")

chain = options.get_options_chain("^GSPC")

data = pd.DataFrame(chain['calls'])
data1 = pd.DataFrame(chain['puts'])

data.to_csv('calls.csv')
data1.to_csv('puts.csv')


plt.show()
print(chain["calls"].head())
