from keras.layers import Dense, Input, Dropout

class ForecastNet(object):

    def __init__(self, input_shape, output_dim, hidden_dims=[512, 256, 128],
                    activation='relu', output_activation=None, pdrop=0.2):
        x = Input(shape=(input_shape))
        h1 = Dense(hidden_dims[0],
                   kernel_initializer='random_normal',
                    bias_initializer='random_normal',
                    activation=activation)
        do1 = Dropout(pdrop)
        h2 = Dense(hidden_dims[1],
                   kernel_initializer='random_normal',
                   bias_initializer='random_normal',
                   activation=activation)
        do2 = Dropout(pdrop)
        h3 = Dense(hidden_dims[2],
                           kernel_initializer='random_normal',
                            bias_initializer='random_normal',
                            activation=activation)
        do3 = Dropout(pdrop)
        h4 = Dense(output_dim, activation=output_activation)

    def call(self, x):
        x = h1(x)
        x = do1(x)
        x = h2(x)
        x = do2(x)
        x = h3(x)
        x = do3(x)
        return h4(x)