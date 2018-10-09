""" Module to read in csv file and options from the user, and create a dictionary with everything needed for 
    subsequent neural network training """

necessary_parameters = [
                       'x', 'y', 'epochs', 'patience', 'train_size', 
                       'test_size', 'arch_list', 'lam_reg'
                       ]

param_dict = {
             'target_column': {
                              'default': -1,
                              'type': int
                              },
             'seed': {
                     'default': 42,
                     'type': int
                     },
             'epochs': {
                       'default': 1000,
                       'type': int
                       },
             'patience': {
                         'default': 100,
                         'type': int
                         },
             'train_size': {
                           'default': 0.75,
                           'type': float
                           },
             'test_size': {
                          'default': 0.25,
                          'type': float
                          },
             'lam_reg': {
                        'default': 1.,
                        'type': float
                        }
             }

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
        return_dict['OK'] = False
        return_dict['message'] = 'Unable to read data file:'
        return return_dict

    data_shape = file_data.shape
    n_features = data_shape[1] - 1
    n_samples = data_shape[0]

    if '-arch_list' not in sys_argv:
        return_dict['OK'] = False
        return_dict['message'] = "Architecture not specified"
        return return_dict

    arch_list = []
    index = sys_argv.index('-arch_list')
    while True:
        try:
            arch_list.append(int(sys_argv[index + 1]))
            index += 1
        except:
            break

    if len(arch_list) == 0:
        return_dict['OK'] = False
        return_dict['message'] = (
                                 "Architecture must be specified " +
                                 "as a sequence of integers after " +
                                 "-arch_list."
                                 )
        return return_dict

    return_dict['arch_list'] = arch_list


    for run_param in param_dict:
        argument = '-' + run_param
        param_type = param_dict[run_param]['type']
        if argument in sys_argv:
            try:
                index = sys_argv.index(argument)
                value = param_type(sys_argv[index + 1])
                return_dict[run_param] = value
            except:
                return_dict['OK'] = False
                return_dict['message'] = (
                                         'Value of ' + run_param + 
                                         'must be of ' + str(param_type) +
                                         ' and provided immediately ' +
                                         'after ' + argument + '.'
                                         )
                return return_dict
        else:
            return_dict[run_param] = param_dict[run_param]['default']

    try:
        y = file_data[:, return_dict['target_column']]
    except IndexError:
        return_dict['OK'] = False
        return_dict['message'] = (
                                 return_dict['target_column'] + 
                                 ' is not a valid column.'
                                 )
        return return_dict

    mask = (file_data == file_data)
    mask[:, return_dict['target_column']] = False
    x = file_data[mask].reshape((n_samples, n_features))

    return_dict['x'] = x
    return_dict['y'] = y

    return_dict['necessary_parameters'] = necessary_parameters

    return return_dict

