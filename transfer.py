from keras import models
from keras import layers
from keras import optimizers
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

stock = ['AAPL','GOOGL','NFLX','SPY']

scale = 1000

for st in stock:
    data = pd.read_csv('Data/'+st+'calls.csv')
    xs_list = data['Strike'].tolist()/scale+xs_list
    ys_list = data['Maturity'].tolist()/scale+ys_list
    zs_list = data['Implied Volatility'].tolist()/scale+zs_list

data = pd.read_csv('Data/TSLAcalls.csv')
tslax = data['Strike'].tolist()/scale
tslay = data['Maturity'].tolist()/scale
tslaz = data['Implied Volatility'].tolist()/scale

tslat = np.column_stack((tslax, tslay))

temp = np.column_stack((xs_list, ys_list))

epochs = 100

model = models.Sequential()
model.add(layers.Dense(50, activation='relu', input_dim=(2)))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

x_train, x_test, y_train, y_test = train_test_split(tslat, tslaz, test_size = 0.8)

model.compile(optimizer=optimizers.Adam(),loss='mean_squared_error',metrics=['mae'])
history = model.fit(temp, zs_list,epochs=epochs,batch_size=10)

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
