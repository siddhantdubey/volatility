from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs_list = []
ys_list = []
zs_list = []

n = 100
datess = ['03/06/2020','03/13/2020','03/20/2020','03/27/2020','04/03/2020','04/17/2020','05/15/2020','06/19/2020','07/17/2020','08/21/2020','09/18/2020','10/16/2020','01/15/2021','03/19/2021','06/18/2021','09/17/2021','01/21/2022','03/18/2022','06/17/2022']
f_datess = []
for i in range(len(datess)):
    d = datess[i]
    temp = d[6:] + '-' + d[:2] + '-' + d[3:5]
    f_datess.append(temp)
# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for j in range(len(f_datess)):
    for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
        data = pd.read_csv('Data/calls' + f_datess[j] + '.csv')
        data = data[data['Implied Volatility'] != 0]
        xs = data['Strike'].tolist()
        ys = data['Maturity'].tolist()
        zs = data['Implied Volatility'].tolist()
        print(type(xs))
        xs_list = xs_list + xs
        ys_list = ys_list + ys
        zs_list = zs_list + zs

ax.scatter(xs_list, ys_list, zs_list)
print(len(xs_list))
print(len(ys_list))
print(len(zs_list))

ax.set_xlabel('Stike')
ax.set_ylabel('Maturity')
ax.set_zlabel('Implied Volatility')

plt.show()

