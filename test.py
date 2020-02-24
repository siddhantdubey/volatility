import pandas as pd 
import numpy as np 

data = pd.read_csv('calls.csv')

for x in range(len(data)):
    print((data.iloc[[x]]['Strike']))