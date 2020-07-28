# COVID-19 Weekly Forecast in US

## Basic project description:
  Purpose is to obtain daily updated data on US Covid-19 numbers, which will be processed along with data from previous dates. The model yields prediction on next 7 days on whether a US state has a potential to be in a risk of having a high "incident rate" or not.

  Statistical Analysis/Methods:
  * EDA on which feature values and target value are appropriate
  * Assessed using possible feature variables - 'Confirmed', 'Active', 'Deaths', 'People_Tested', and 'Testing_Rate' (based on Johns Hopkins CSSE data).
  * Multiple Linear Regression model to obtain coefficients, and thus distinguish data points which are significantly off from the mean values
  *  R-squared values to compare which model should be used
  * Timeseries train/test split for model evaluation


## Updates:

* July 26, 2020: Organizing code and fixing bugs; worked up to filtering process
* July 25, 2020: Fixed bugs, data transfer errors, and prepped for data analysis
* July 24, 2020: Updated workload on cleaning data and pre-aggregation (process: ts diff > mean agg > outliers > classify)
* July 23, 2020: Archive old files and update new workload; started EDA
