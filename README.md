# COVID-19 Weekly Forecast in US

## Basic project description:
  Purpose is to classify "in risk" or "not in risk" for the following week on new cases for each state. Utilize previous model forecasts to train a new neural network and thus obtain higher prediction accuracy.

  Statistical Analysis/Methods:
  * EDA on which feature values and target value are appropriate
  * Using previous CDC weekly case forecasts, aggregate all model predictions and measure errors
  * Use gradients to update weights on training
  * & more


## Updates:

* ~Sep 12, 2020: Uploaded recent workload on building multi-layer perceptron network
* ~Sep 06, 2020: Updated new training methods and data: use previous forecasts to train for higher accuracy
* ~Aug 06, 2020: Worked on scaling methods and model evaluation. Figuring out how to deal with 56 states' dataset for each of 100 total date data, in order to produce the result I was visioning.
* July 28, 2020: Separated and cleaned up data pre-processing & analysis part of the work. Planned to assess log linreg rather than multi-linreg.
* July 27, 2020: Completed up to evaluation of linear regression models
* July 26, 2020: Organizing code and fixing bugs; worked up to filtering process
* July 25, 2020: Fixed bugs, data transfer errors, and prepped for data analysis
* July 24, 2020: Updated workload on cleaning data and pre-aggregation (process: ts diff > mean agg > outliers > classify)
* July 23, 2020: Archive old files and update new workload; started EDA
