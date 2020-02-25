# Volatility Surfaces

The analysis conducted in this repository is currently conducted on $TSLA options. 

We have so far attempted Random Forest Regression, Gradient Boosted Regression, Kriging, Nearest Neighbor, directly imported from the [scikit learn python package](https://scikit-learn.org/) and the [pykrige package](https://github.com/GeoStat-Framework/PyKrige). 

## Random Forest Regression

### Output

![Random Forest Regression](https://github.com/siddhantdubey/volatility/blob/master/Graphics/FitImages/forestregression.png?raw=true)



## Gradient Boosted Regression

### Output

![Random Forest Regression](https://github.com/siddhantdubey/volatility/blob/master/Graphics/FitImages/gbdregression.png?raw=true)



## Voter Regression

### Output

![Voter Regression](https://github.com/siddhantdubey/volatility/blob/master/Graphics/FitImages/votingregression.png?raw=true)



## Kriging

The following output is fairly bad, this is most likely due to poor implementation. Kriging is a technique taken from geo-statistics.

### Output

![Krigging Analysis](https://github.com/siddhantdubey/volatility/blob/master/Graphics/FitImages/snaps__top_kriging.png?raw=true)

## MLP Regression

This uses Multi Layer Perceptron method, the following was done with a hidden layer size of 400, trained over 450 epochs.

![MLP Regression](https://github.com/siddhantdubey/volatility/blob/master/Graphics/FitImages/mlpregression.png?raw=true)

