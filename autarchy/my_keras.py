""" Module with methods for constructing dense neural networks, for growing layers, ... """

from tensorflow import keras

class PrintDot(keras.callbacks.Callback):
    """ Copied from the tutorial at 
    https://www.tensorflow.org/tutorials/keras/basic_regression """
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


def model_from_architecture(
                           arch_list, lam_reg, 
                           activation='sigmoid',
                           initializer='normal',
                           loss='mean_squared_error',
                           optimizer='adam',
                           regularizer='l2'
                           ):
    """Return a dense neural network
    
    arch_list: [input_dim, hidden_layer_1_neurons, ..., output_dim]
    lam_reg: l2 regulator for each layer
    activation: activation function for each neuron outside the final layer
    intializer: function to intialize the weights
    regularizer: either 'l1' or 'l2'-- selection of weight regularization
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras import regularizers

    assert len(arch_list) > 2

    assert regularizer.upper() in ['L1', 'L2']

    if regularizer.upper() == 'L1':
        regularizer = regularizers.l1
    else:
        regularizer = regularizers.l2

    input_dim = arch_list[0]
    output_dim = arch_list[-1]
    
    my_model = Sequential()
    
    # first hidden layer
    my_model.add(
                Dense(
                     arch_list[1],
                     kernel_initializer=initializer,
                     activation=activation,
                     input_dim=input_dim,
                     kernel_regularizer=regularizer(lam_reg)
                     )
                )
    
    # subsequent hidden layers
    for n_neurons in arch_list[2:-1]:
        my_model.add(
                    Dense(
                         n_neurons,
                         kernel_initializer=initializer,
                         activation = activation,
                         kernel_regularizer=regularizer(lam_reg)
                         )
                    )
        
    # output layer (regression, so no activation)
    my_model.add(
                Dense(
                     output_dim,
                     kernel_initializer=initializer,
                     kernel_regularizer=regularizer(lam_reg)
                     )
                )
    
    my_model.compile(loss=loss, optimizer=optimizer)
    return my_model