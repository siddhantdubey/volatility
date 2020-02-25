from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

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
        data = pd.read_csv('calls' + f_datess[j] + '.csv')
        data = data[data['Implied Volatility'] != 0]
        xs = data['Strike'].tolist()
        ys = data['Maturity'].tolist()
        zs = data['Implied Volatility'].tolist()
        xs_list = xs_list + xs
        ys_list = ys_list + ys
        zs_list = zs_list + zs

temp = np.column_stack((xs_list, ys_list))
# regr = MLPRegressor(hidden_layer_sizes=100, activation='relu') MLP Performs significantly worse than RandomForest
# regr = RandomForestRegressor(max_depth=200, random_state=0)
regr = GradientBoostingRegressor()
# regr = VotingRegressor(estimators=[('gb', reg2), ('rf', reg1)])
x_train, x_test, y_train, y_test = train_test_split(temp, zs_list, test_size = 0.01)

regr.fit(x_train, y_train)

ymin = 10
ymax = 800
yinc = 10

xmin = 350
xmax = 2000
xinc = 25

X = np.arange(xmin, xmax, xinc)
Y = np.arange(ymin, ymax, yinc)

X, Y = np.meshgrid(X, Y)

x = np.ndarray.flatten(X)
y = np.ndarray.flatten(Y)
xtest = np.stack((x, y), axis=-1)



predict = regr.predict(xtest)

# pred = np.ndarray.squeeze(predict, axis=1)
pred = np.reshape(predict, (int((ymax-ymin)/yinc),int((xmax-xmin)/xinc)))

print(pred)

fig1 = plt.figure()
ax1 = fig1.gca(projection='3d')

# Make data.
# Plot the surface.
surf1 = ax1.plot_surface(X, Y, pred, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax1.set_zlim(0, 250)
ax1.zaxis.set_major_locator(LinearLocator(10))
ax1.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig1.colorbar(surf1, shrink=0.5, aspect=5)

plt.savefig('images/gbdregression.png')

# fig = plt.figure()
# ax = fig.gca(projection='3d')

# # Make data.
# X = np.arange(0, 2, 0.1)
# Y = np.arange(0, 2, 0.1)
# x, y = np.meshgrid(X, Y)
# Z = .3*(.1*x**3+.2*y**2+.1*x*y +.5*x+.5*y)-.5

# # Plot the surface.
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)

# # Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()