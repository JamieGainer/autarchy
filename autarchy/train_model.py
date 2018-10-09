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



EPOCHS = 1000
patience = 100

train_size = 0.75
test_size  = 0.25



x, y = [run_dictionary[label] for label in ['x', 'y']]
EPOCHS, patience = [run_dictionary[label] for label in ['EPOCHS', 'patience']]


(
x_train, 
x_test,
y_train,
y_test
) = train_test_split(
                    x, y, train_size=train_size,
                    test_size=test_size, 
                    random_state=run_dictionary['seed']
                    )

scaler = RobustScaler()
scaled_x_train = scaler.fit_transform(x_train)

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=500)
arch_list = [13,1000,1000,1]
lam_reg = 0.1

arch_list = [13,100,60,30,20,10,1]
my_model = my_keras.model_from_architecture(arch_list, lam_reg)
print(arch_list, lam_reg)

history = my_model.fit(
                      scaled_x_train, y_train, epochs=EPOCHS,
                      validation_split=0.1, 
                      verbose=0,
                      callbacks=[my_keras.PrintDot(), early_stop]
                      )
        
val_loss = history.history['val_loss'][-1]
train_loss = history.history['loss'][-1]
scaled_x_test = sx_testr.transform(x_test)
test_loss = my_model.predict(scaled_x_test)

print('\nval_loss:', val_loss)
print('train_loss:', train_loss)
print('test_loss:', test_loss)
print('time:', time.time() - start_time)

