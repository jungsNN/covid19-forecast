# COVID-19 Weekly Forecast in US

<hr></hr>
## Basic project description:
  Purpose is to classify and predict whether a state will be at status, `reopened`, `in risk` or `closed` the following days (7 day prediction). `COVID-19 United States Reopen and Shut Down Status by State` by rearc and `Coronavirus (Covid-19) Data Hub` by Tableau.

  __Dataset:__

  * Most of feature variables (60,000+ records) accessed from rearc data:
    - Aggregated daily means for all continuous variables by `county_fips_number`

  * Target variables from Tableau's Covid-19 Data Hub:
    - `reopened`
    - `in risk`
    - `closed`

  __Project Division:__

  * EDA on which feature values and target value are appropriate for project purpose
  * Feature conversion and engineering
  * Merging feature dataset and target dataset
  * Train/test split
  * Building a Recurrent Neural Network architecture with LSTM using Pytorch

<hr></hr>
## References

* [COVID-19 United States Reopen and Shut Down Status by State](https://github.com/rearc-data/nyt-states-reopen-status-covid-19)
  - Target variables classified into `reopened`, `in risk` and `closed`
  - Other feature variables considered into training: `population`
* [Coronavirus (COVID-19) Data Hub AWS Description](https://console.aws.amazon.com/dataexchange/home?region=us-east-1#/subscriptions/prod-ed6ulhryl6cjs)
  - Based on [European Centre for Disease Prevention and Control](https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide), [The New York Times](https://github.com/nytimes/covid-19-data) & [Public Health Agency of Canada](https://www.canada.ca/en/public-health/services/diseases/2019-novel-coronavirus-infection.html?topic=tilelink)
  - Extracted most of project's feature variables from the source
  - More than 60,000 records used for training

<hr></hr>
## Updates:

* Oct 30, 2020: Added a bit more thorough EDA section prior to building a neural network; started building a neural network (details TBA)
* Oct 26, 2020: Updated EDA and completed up to splitting into train, val, test dataloaders for training. Need to fix fetching datasets via requests to sources. Updated python "helper" module, `co_helper`, which includes python functions.
* ~Sep 12, 2020: Uploaded recent workload on building multi-layer perceptron network
* ~Sep 06, 2020: Updated new training methods and data: use previous forecasts to train for higher accuracy
* ~Aug 06, 2020: Worked on scaling methods and model evaluation. Figuring out how to deal with 56 states' dataset for each of 100 total date data, in order to produce the result I was visioning.
* July 28, 2020: Separated and cleaned up data pre-processing & analysis part of the work. Planned to assess log linreg rather than multi-linreg.
* July 27, 2020: Completed up to evaluation of linear regression models
* July 26, 2020: Organizing code and fixing bugs; worked up to filtering process
* July 25, 2020: Fixed bugs, data transfer errors, and prepped for data analysis
* July 24, 2020: Updated workload on cleaning data and pre-aggregation (process: ts diff > mean agg > outliers > classify)
* July 23, 2020: Archive old files and update new workload; started EDA
