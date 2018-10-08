""" Module to read in csv file and options from the user, and create a dictionary with everything needed for 
    subsequent neural network training """


def read_data_from_command_line_args(sys_argv):

    import numpy as np

    return_dict = {'OK': True}

    try:
        input_file_name = sys_argv[1]
    except:
        return_dict['OK'] = False
        return_dict['message'] = 'Need csv file name as first input.'
        return return_dict

    try:
        file_data = np.genfromtxt(
                                 input_file_name, delimiter=',', 
                                 skip_header=1
                                 )
    except:
        return_dict['OK']
        return_dict['message'] = 'Unable to read data file:'
        return return_dict

    data_shape = file_data.shape
    n_features = data_shape[1] - 1
    n_samples = data_shape[0]

    param_dict = {
                 'target_column': -1,
                 'seed': 42
                 }

    for run_param in param_dict:
        argument = '-' + run_param
        if argument in sys.argv:
            try:
                index = sys.argv.index(argument)
                value = int(sys.argv[index + 1])
                param_dict[run_param] = value
            except:
                return_dict['OK'] = False
                return_dict['message'] = (
                                         'Value of ' + argument + 
                                         'must be integer provided ' +
                                         'after' + argument + '.'
                                         )
                return return_dict

    return_dict.update(param_dict)

    try:
        y = file_data[:, param_dict['target_column']]
    except IndexError:
        return_dict['OK'] = False
        return_dict['message'] = (
                                 param_dict['target_column'] + 
                                 ' is not a valid column.'
                                 )
        return return_dict

    mask = (file_data == file_data)
    mask[:, param_dict['target_column']] = False
    x = file_data[mask].reshape((n_samples, n_features))

    return_dict['x'] = x
    return_dict['y'] = y

    return return_dict

