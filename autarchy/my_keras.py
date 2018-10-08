""" Module with methods for constructing dense neural networks, for growing layers, ... """

def model_from_architecture(
                           arch_list, lam_reg, 
                           activation='sigmoid',
                           initializer='normal',
                           loss='mean_squared_error',
                           optimizer='adam'
                           ):
    """Return a dense neural network
    
    arch_list: [input_dim, hidden_layer_1_neurons, ..., output_dim]
    lam_reg: l2 regulator for each layer
    activation: activation function for each neuron outside the final layer
    intializer: function to intialize the weights
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras import regularizers

    assert len(arch_list) > 2
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
                     kernel_regularizer=regularizers.l2(lam_reg)
                     )
                )
    
    # subsequent hidden layers
    for n_neurons in arch_list[2:-1]:
        my_model.add(
                    Dense(
                         n_neurons,
                         kernel_initializer=initializer,
                         activation = activation,
                         kernel_regularizer=regularizers.l2(lam_reg)
                         )
                    )
        
    # output layer (regression, so no activation)
    my_model.add(
                Dense(
                     output_dim,
                     kernel_initializer=initializer,
                     kernel_regularizer=regularizers.l2(lam_reg)
                     )
                )
    
    my_model.compile(loss=loss, optimizer=optimizer)
    return my_model