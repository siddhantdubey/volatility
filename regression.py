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
# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].

data = pd.read_csv('Data/SPYcalls.csv')
xs_list = data['Strike'].tolist()
ys_list = data['Maturity'].tolist()
zs_list = data['Implied Volatility'].tolist()

temp = np.column_stack((xs_list, ys_list))
regr = MLPRegressor(hidden_layer_sizes=(50,50), activation='relu', max_iter=1000) #MLP Performs significantly worse than RandomForest
regrf = RandomForestRegressor(n_estimators = 500, random_state=0)
# reg2 = GradientBoostingRegressor()
reg1 = VotingRegressor(estimators=[('ml', regr), ('rf', regrf)], weights=[.5,.5])
x_train, x_test, y_train, y_test = train_test_split(temp, zs_list, test_size = 0.01)

reg1.fit(x_train, y_train)

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



predict = reg1.predict(xtest)

# pred = np.ndarray.squeeze(predict, axis=1)
pred = np.reshape(predict, (int((ymax-ymin)/yinc),int((xmax-xmin)/xinc)))

print(pred)

fig1 = plt.figure(figsize=(10, 5))
ax1 = fig1.gca(projection='3d')

# Make data.
# Plot the surface.
surf1 = ax1.plot_surface(X, Y, pred, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax1.set_zlim(0, 300)
ax1.zaxis.set_major_locator(LinearLocator(10))
ax1.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig1.colorbar(surf1, shrink=0.5, aspect=5)

plt.savefig('Graphics/FitImages/randomforest.png')
plt.show()
