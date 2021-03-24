# Volatility Surfaces

The analysis conducted in this repository is currently conducted on $TSLA options. 

We have so far attempted Random Forest Regression, Gradient Boosted Regression, Kriging, Nearest Neighbor, directly imported from the [scikit learn python package](https://scikit-learn.org/) and the [pykrige package](https://github.com/GeoStat-Framework/PyKrige). 

## Random Forest Regression

### Output

![Random Forest Regression on $TSLA Options](https://user-images.githubusercontent.com/23504484/112371180-b2669000-8cb4-11eb-9d0e-2a9a1f87f60e.png)
https://github.com/siddhantdubey/volatility/blob/master/Graphics/FitImages/forestregression.png?raw=true)



## Gradient Boosted Regression

### Output

![GBD Regression](https://user-images.githubusercontent.com/23504484/112371282-d0cc8b80-8cb4-11eb-8866-6b51dbd14d90.png)




## Voter Regression

### Output

![Voter Regression](https://user-images.githubusercontent.com/23504484/112371312-db872080-8cb4-11eb-8a2e-977ccb5d7e4c.png)


## Kriging

The following output is fairly bad, this is most likely due to poor implementation. Kriging is a technique taken from geo-statistics.

### Output

![Kriging Image](https://user-images.githubusercontent.com/23504484/112371380-eb9f0000-8cb4-11eb-84b2-a25f9e7cd1be.png)


## MLP Regression

This uses the Multi Layer Perceptron method, the following was done with a hidden layer size of 400, trained over 450 epochs.

![MLP Regression](https://user-images.githubusercontent.com/23504484/112371421-f9548580-8cb4-11eb-9b1d-a8158fab83a4.png)


