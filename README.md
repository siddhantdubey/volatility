# Volatility Surfaces

The analysis conducted in this repository is currently conducted on $TSLA options. 

We have so far attempted Random Forest Regression, Gradient Boosted Regression, Kriging, Nearest Neighbor, directly imported from the [scikit learn python package](https://scikit-learn.org/) and the [pykrige package](https://github.com/GeoStat-Framework/PyKrige). 

## Random Forest Regression

### Output

![Random Forest Regression](C:\Users\siddh\Documents\CLASS-X\Finance\ProjectReal\BigData\Graphics\FitImages\forestregression.png)

## Gradient Boosted Regression

### Output

![Random Forest Regression](C:\Users\siddh\Documents\CLASS-X\Finance\ProjectReal\BigData\Graphics\FitImages\gbdregression.png)

## Voter Regression

### Output

![Voter Regression](C:\Users\siddh\Documents\CLASS-X\Finance\ProjectReal\BigData\Graphics\FitImages\votingregressor.png)



## Kriging

The following output is fairly bad, this is most likely due to poor implementation. Kriging is a technique taken from geo-statistics.

### Output

![](C:\Users\siddh\Documents\CLASS-X\Finance\ProjectReal\BigData\Graphics\FitImages\snaps__bot_kriging.png)

## MLP Regression

This uses Multi Layer Perceptron method, the following was done with a hidden layer size of 400, trained over 450 epochs.

![MLP Regression](C:\Users\siddh\Documents\CLASS-X\Finance\ProjectReal\BigData\Graphics\FitImages\mlpregression.png)



