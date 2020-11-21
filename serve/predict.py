import pandas as pd
import numpy as np
import datetime
import tensorflow as tf


class Predict(object):

    def __init__(self, query):
        self.query = query
        self.input_variables = ['date','cases', 'deaths', 'fips']
        self.output_variables = ['total_open', 'cases', 'deaths', 'status']
        self.updated_csv_filepath = './data/today_csv.csv'
        self.model_filepath = "./data/assets/forecast_trained.h5"
        self.today_date = datetime.date.today()
        self.next_week_date = self.today_date + datetime.timedelta(7)
        # variable placeholders
        self.fips_list = np.load('./data/x_fips.npy').tolist()
        self.new_fips_list = []
        self.max_val = []

    def normalize(self, data):
        data_ = data.copy()
        for v, var_name in enumerate(self.output_variables[:-1]):
            self.max_val.append(data[var_name].max())
            data_.loc[:, var_name] = data[var_name] / self.max_val[v]
        return data_

    def preprocess_data(self):
        """
        Preprocess csv data for prediction input.
        param
        -----
        url(str): URL to NY Times `us-states.csv` raw GitHub data
        dummy_fips(numpy array): filtered 51 fips
        return
        ------
        x(numpy array): input features for prediction
        """
        today_data = pd.read_csv(self.updated_csv_filepath, names=self.input_variables)
        today_data = today_data.set_index('fips').drop([f for f in list(today_data.fips.unique())
                                                        if f not in self.fips_list]).reset_index()
        self.new_fips_list.append([fip for fip in today_data.fips])
        # normalize
        today_data = self.normalize(today_data)
        fips_dummy = tf.one_hot(today_data.fips, len(self.new_fips_list))
        # concatenating all three feature arrays together; shape should be (51, 53)
        self.get_predictions(np.concatenate((fips_dummy, np.array(today_data.loc[:, ['cases', 'deaths']])), axis=1))

    def get_predictions(self, input_arr):
        model = tf.keras.models.load_model(self.model_filepath)
        result = model.predict(input_arr)
        self.serialize_prediction(result)

    def serialize_prediction(self, prediction):
        """
        Predicted data format:
        [total_open, cases, deaths, status]
        Params
        -----
        var_max(list): list of each variable's max value in the
            order above; excludes `status`
        fips_list(list): fips list in the order corresponding to the
            fitted train dataset
        """
        # creating a dataframe of predictions
        outcome_data = pd.DataFrame(prediction,
                                    columns=self.output_variables,
                                    index=self.new_fips_list)
        # scaling back numerical values using their saved max
        outcome_data_ = outcome_data.copy()
        for i in range(len(self.max_val)):
            outcome_data_[self.output_variables[i]] = outcome_data[
                self.output_variables[i]].apply(lambda x: x * self.max_val[i])
        # TODO: CONVERT FIPS INTO STATE NAMES AND APPEND TO PREVIOUS PRED DATA
        # create a column of correspondingn date
        date_col = [self.next_week_date.strftime("%Y-%m-%d")]*len(outcome_data_)
        outcome_data_['date'] = date_col
        # resetting index to reorder the columns ['date', 'fips', output_variables]
        self.respond_query(outcome_data_.reset_index().set_index('date').reset_index())

    def respond_query(self, serialized_data):
        # first, pulling up previous predictions
        previous_predicts = pd.read_csv('previous_predicts.csv',
                                        names=['date', 'fips', 'total_open', 'cases', 'deaths', 'status'])
        # append all of the prediction data
        previous_predicts = pd.concat([previous_predicts.set_index('date'), serialized_data])
