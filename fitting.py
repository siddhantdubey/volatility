import sys
import os.path
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import numpy as np
from os import path
from pykrige.ok import OrdinaryKriging
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from scipy.interpolate import griddata as gd
from mpl_toolkits.mplot3d import axes3d
from scipy.interpolate import griddata as gd

stock = input("Input Stock Ticker: ")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs_list = []
ys_list = []
zs_list = []

if(path.exists('Graphics/FitImages/' + stock)):
    pass
else:
    os.mkdir('Graphics/FitImages/' + stock)

n = 100
# datess = ['03/06/2020','03/13/2020','03/20/2020','03/27/2020','04/03/2020','04/17/2020','05/15/2020','06/19/2020','07/17/2020','08/21/2020','09/18/2020','10/16/2020','01/15/2021','03/19/2021','06/18/2021','09/17/2021','01/21/2022','03/18/2022','06/17/2022']
# f_datess = []
# for i in range(len(datess)):
#     d = datess[i]
#     temp = d[6:] + '-' + d[:2] + '-' + d[3:5]
#     f_datess.append(temp)
# # For each set of style and range settings, plot n random points in the box
# # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
# for j in range(len(f_datess)):
#     for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
#         data = pd.read_csv('Data/calls' + f_datess[j] + '.csv')
#         data = data[data['Implied Volatility'] != 0]
#         xs = data['Strike'].tolist()
#         ys = data['Maturity'].tolist()
        # zs = data['Implied Volatility'].tolist()
#         xs_list = xs_list + xs
#         ys_list = ys_list + ys
#         zs_list = zs_list + zs


data = pd.read_csv('Data/' + stock + 'calls.csv')

xs_list = data['Strike'].tolist()
ys_list = data['Maturity'].tolist()
zs_list = data['Implied Volatility'].tolist()
epic = []

for x in range(len(xs_list)):
    listy=[]
    listy.append(xs_list[x])
    listy.append(ys_list[x])
    listy.append(zs_list[x])
    epic.append(listy)

epic = np.array(epic)
real_data = epic[:3000]
print(real_data)
interpolationmethod = 'cubic'
p = 2
extrapolation_interval = 10


def main():
    extrapolation_spots = get_plane(0, 2000, 0, 300, extrapolation_interval)
    nearest_analysis(extrapolation_spots)
    kriging_analysis(extrapolation_spots)

def get_plane(xl, xu, yl, yu, i):
    xx = np.arange(xl, xu, i)
    yy = np.arange(yl, yu, i)
    extrapolation_spots = np.zeros((len(xx) * len(yy), 2))
    count = 0
    for i in xx:
        for j in yy:
            extrapolation_spots[count, 0] = i
            extrapolation_spots[count, 1] = j
            count += 1
    return extrapolation_spots

def nearest_analysis(extrapolation_spots):
    top_extra = extrapolation(real_data, extrapolation_spots, method='nearest')
    bot_extra = extrapolation(real_data, extrapolation_spots, method='nearest')
    gridx_top, gridy_top, gridz_top = interpolation(top_extra)
    plot(real_data, gridx_top, gridy_top, gridz_top, method='snaps',
            title='_top_nearest')
    gridx_bot, gridy_bot, gridz_bot = interpolation(bot_extra)
    plot(real_data, gridx_bot, gridy_bot, gridz_bot, method='snaps',
            title='_bot_nearest')

    plot(np.concatenate((real_data, real_data)), [gridx_top, gridx_bot],
              [gridy_top, gridy_bot],
              [gridz_top, gridz_bot], method='snaps', title='_both_nearest',
            both=True)
def nearest_neighbor_interpolation(data, x, y, p=0.5):
    """
    Nearest Neighbor Weighted Interpolation
    http://paulbourke.net/miscellaneous/interpolation/
    http://en.wikipedia.org/wiki/Inverse_distance_weighting

    :param data: numpy.ndarray
        [[float, float, float], ...]
    :param p: float=0.5
        importance of distant samples
    :return: interpolated data
    """
    n = len(data)
    vals = np.zeros((n, 2), dtype=np.float64)
    distance = lambda x1, x2, y1, y2: (x2 - x1)**2 + (y2 - y1)**2
    for i in range(n):
        vals[i, 0] = data[i, 2] / (distance(data[i, 0], x, data[i, 1], y))**p
        vals[i, 1] = 1          / (distance(data[i, 0], x, data[i, 1], y))**p
    z = np.sum(vals[:, 0]) / np.sum(vals[:, 1])
    return z

def kriging_analysis(extrapolation_spots):
    top_extra = extrapolation(real_data, extrapolation_spots, method='kriging')
    bot_extra = extrapolation(real_data, extrapolation_spots, method='kriging')
    gridx_top, gridy_top, gridz_top = interpolation(top_extra)
    plot(real_data, gridx_top, gridy_top, gridz_top, method='snaps',
            title='_top_kriging')
    gridx_bot, gridy_bot, gridz_bot = interpolation(bot_extra)
    plot(real_data, gridx_bot, gridy_bot, gridz_bot, method='snaps',
            title='_bot_kriging')

    plot(np.concatenate((real_data, real_data)), [gridx_top, gridx_bot],
              [gridy_top, gridy_bot],
              [gridz_top, gridz_bot], method='snaps', title='_both_kriging',
            both=True)
def kriging(data, extrapolation_spots):
    """
    https://github.com/bsmurphy/PyKrige

    NOTE: THIS IS NOT MY CODE

    Implementing a kriging algorithm is out of the scope of this homework

    Using a library. See attached paper for kriging explanation.
    """
    gridx = np.arange(0.0, 2000, 10)
    gridy = np.arange(0.0, 2000, 10)
    # Create the ordinary kriging object. Required inputs are the X-coordinates of
    # the data points, the Y-coordinates of the data points, and the Z-values of the
    # data points. If no variogram model is specified, defaults to a linear variogram
    # model. If no variogram model parameters are specified, then the code automatically
    # calculates the parameters by fitting the variogram model to the binned
    # experimental semivariogram. The verbose kwarg controls code talk-back, and
    # the enable_plotting kwarg controls the display of the semivariogram.
    OK = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2], variogram_model='spherical',
                                 verbose=False, nlags=100)
    
    # Creates the kriged grid and the variance grid. Allows for kriging on a rectangular
    # grid of points, on a masked rectangular grid of points, or with arbitrary points.
    # (See OrdinaryKriging.__doc__ for more information.)
    z, ss = OK.execute('grid', gridx, gridy)
    return gridx, gridy, z, ss

def extrapolation(data, extrapolation_spots, method='nearest'):
    if method == 'kriging':
        xx, yy, zz, ss = kriging(data, extrapolation_spots)

        new_points = np.zeros((len(yy) * len(zz), 3))
        count = 0
        for i in range(len(xx)):
            for j in range(len(yy)):
                new_points[count, 0] = xx[i]
                new_points[count, 1] = yy[j]
                new_points[count, 2] = zz[i, j]
                count += 1
        combined = np.concatenate((data, new_points))
        return combined

    if method == 'nearest':
        new_points = np.zeros((len(extrapolation_spots), 3))
        new_points[:, 0] = extrapolation_spots[:, 0]
        new_points[:, 1] = extrapolation_spots[:, 1]
        for i in range(len(extrapolation_spots)):
            new_points[i, 2] = nearest_neighbor_interpolation(data,
                                    extrapolation_spots[i, 0],
                                    extrapolation_spots[i, 1], p=p)
        combined = np.concatenate((data, new_points))
        return combined


def interpolation(data):
    gridx, gridy = np.mgrid[0:150:50j, 0:150:50j]
    gridz = gd(data[:, :2],data[:, 2], (gridx, gridy),
                method=interpolationmethod)
    return gridx, gridy, gridz


def plot(data, gridx, gridy, gridz, method='rotate', title='nearest', both=False):
    def update(i):
        ax.view_init(azim=i)
        return ax,

    if method == 'rotate':
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='3d')

        ax.plot_wireframe(gridx, gridy, gridz, alpha=0.5)
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='red')

        animation.FuncAnimation(fig, update, np.arange(360 * 5), interval=1)
        plt.savefig('Graphics/FitImages/' + stock + '/rotate_{}.gif'.format(title))

    elif method== 'snaps':
        fig = plt.figure(figsize=(10, 10))
        angles = [45, 120, 220, 310]

        if both:
            list = [1, 2, 3, 4]
            for i in range(4):
                ax = fig.add_subplot(1, 1,1, projection='3d')
                ax.plot_wireframe(gridx[0], gridy[0], gridz[0], alpha=0.5)
                ax.plot_wireframe(gridx[1], gridy[1], gridz[1], alpha=0.5)
                ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='red')
                ax.view_init(azim=angles[i])
        else:
            for i in range(4):
                ax = fig.add_subplot(1, 1,1, projection='3d')
                ax.plot_wireframe(gridx, gridy, gridz, alpha=0.5)
                ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='red')
                ax.view_init(azim=angles[i])

        plt.savefig('Graphics/FitImages/' + stock + '/snaps_{}.png'.format(title))

    elif method == 'contour':
        fig = plt.figure()
        ax = fig.add_axes([2000, 2000, 2000, 2000], projection='3d')

        ax.plot_wireframe(gridx, gridy, gridz, alpha=0.5)
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='red')

        ax.contourf(gridx, gridy, gridz, zdir='z', offset=np.min(data[:, 2]), cmap=cm.coolwarm)
        ax.contourf(gridx, gridy, gridz, zdir='x', offset=0, cmap=cm.coolwarm)
        ax.contourf(gridx, gridy, gridz, zdir='y', offset=0, cmap=cm.coolwarm)
        ax.view_init(azim=45)
        plt.savefig('Graphics/FitImages'  + stock + '/contour_{}.png'.format(title))


if __name__ == '__main__':
    sys.exit(main())