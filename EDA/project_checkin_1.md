    1. What data have you gathered, and how did you gather it?
    2. How have you explored the data and what insights have you gained as a result?
    3. Will you be able to answer your question with this data, or do you need to gather more data (or adjust your question)?
    4. What modeling approach are you using to answer your question?


1. I used John's Hopkins CSSE covid-19 daily US data. I cloned the GitHub repo and filtered out
   only the necessary columns I need. The dataset includes daily dates from April 12th 2020 and updates every day on each state's 'confirmed', 'active', 'deaths', 'recovered', 'incident_rate' etc. 

2. I thought the best way to work with the data is to separate the states into two groups by considering the "safe" ranges determined with irq range for target variable for each day, in order to simplify the multi-indexed data into single row timeseries. It seemed to make sense to use changes in daily value, since states differ in magnitude. After creating a new df with diff values, I used .describe() to glimpse over the distribution, and used the percentiles (25th, 75th) to get the iqr and the low,high bounds. I only used the incident_rate column to decipher the "risky" and "normal" values (i.e. state_IR_diff  < low_iqr | > high_iqr). Then, I can categorize the values (in terms of daily/weekly) two groups with binary classification ("risky": 1, "normal": 0). 

   I found that 'incident_rate' is the only variable that seems to directly indicate how "risky" a state is for that day. Other variables doesn't necessarily imply the risk.

3.   I need to find the best method that will deal with variations of the multiple features in predicting the target.

4.  By separating the data into these two groups, I can then use multiple linear regression model in predicting what would be a "normal" range of covid incident rates for each day, as well as weekly terms by doing this upon a 7-day rolling mean.

   With classified target values, the data can be combined together to perform train/test split. Since the data is not large after aggregation, I may have to apply k-folds if necessary. 

