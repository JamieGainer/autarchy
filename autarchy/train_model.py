""" Stuff

    """

from tensorflow import keras
import sys
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

import my_io
import my_keras

start_time = time.time()
run_dict = my_io.read_data_from_command_line_args(sys.argv)
if not run_dict['OK']:
    print(run_dict['message'])
    print('Quitting.')
    quit()

(
x, y, epochs, patience, train_size, test_size, arch_list, lam_reg
) = [run_dict[key] for key in run_dict['necessary_parameters']]

(
x_train, 
x_test,
y_train,
y_test
) = train_test_split(
                    x, y, train_size=train_size,
                    test_size=test_size, 
                    random_state=run_dict['seed']
                    )

scaler = RobustScaler()
scaled_x_train = scaler.fit_transform(x_train)

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=500)

my_model = my_keras.model_from_architecture(arch_list, lam_reg)

for par in run_dict:
    if par not in ['x', 'y', 'necessary_parameters', 'OK']:
        print(par, '\b:\t', run_dict[par])

history = my_model.fit(
                      scaled_x_train, y_train, epochs=epochs,
                      validation_split=0.1, 
                      verbose=0,
                      callbacks=[my_keras.PrintDot(), early_stop]
                      )
        
val_loss = history.history['val_loss'][-1]
train_loss = history.history['loss'][-1]
scaled_x_test = scaler.transform(x_test)
test_loss = my_model.predict(scaled_x_test)

print('\nval_loss:', val_loss)
print('train_loss:', train_loss)
print('test_loss:', test_loss)
print('time:', time.time() - start_time)

