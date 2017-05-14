from __future__ import print_function
import csv
import numpy as np
import argparse
from scipy.stats import pearsonr

def permute_data(data):
    data_len = len(data)
    perm = np.random.permutation(data_len)
    data_perm = data[perm]
    return data_perm



def cross_validation_split(data, labels, crossval_split_index, crossval_total_num_splits, validation_data_ratio = 0.1):
    '''
    Manages cross-validation splits given fixed lists of data/labels
    
    
    <crossval_total_num_splits> directly affects the size of the test set ( it is <size of data-set>/crossval_total_num_splits)
    
    Returns:
    ----------
    
        traindata, valdata, testdata
    
    '''
    assert validation_data_ratio<1 and validation_data_ratio > 0
    assert crossval_split_index < crossval_total_num_splits
    
    N = len(data)
    n_test = int(N*1./crossval_total_num_splits)
    if crossval_split_index == crossval_total_num_splits - 1:
        n_test_full = N - crossval_split_index * n_test
    else:
        n_test_full = n_test
    
    # <valid or train|[@crossval_split_index] test|valid or train>
    
    start_test = crossval_split_index * n_test
    end_test = crossval_split_index * n_test + n_test_full
    testdata = (data[start_test: end_test], labels[start_test: end_test])
    
    rest_data   = np.concatenate((data[:start_test],data[end_test:]), axis=0)
    rest_labels = np.concatenate((labels[:start_test],labels[end_test:]), axis=0)
    
    n_valid   = int(N * validation_data_ratio)
    valdata   = (rest_data[: n_valid], rest_labels[: n_valid])
    traindata = (rest_data[n_valid: ], rest_labels[n_valid: ])
    
    return traindata, valdata, testdata





def read_csv(filename, smile_name, target_name, logp_name=None):
    data = []
    dt = np.dtype('S100, float')
    with open(filename) as file:
        reader = csv.DictReader(file)
        for row in reader:
            data_point=(row[smile_name], float(row[target_name]))
            if logp_name:
                dt = np.dtype('S100, float, float')
                data_point += (float(row[logp_name]),)
            data.append(data_point)

    return np.asarray(data, dtype=dt)


def save_results(file_path, targets, predictions, additional_str=''):
#    data = np.array([targets, predictions])
#    data = data.T
    if file_path[-1]!='/':
        file_path = file_path+'/'
    np.save(file_path+'targets'+additional_str,targets)
    np.save(file_path+'predictions'+additional_str,predictions)
#    f = open(file_path, 'w+')
#    np.savetxt(f, data, delimiter=',', fmt=['%.4f', '%.4f'], header="Target, Prediction", comments="")
#    f.close()


def model_params(s):
    try:
        x, y, z = map(int, s.split(','))
        return x, y, z
    except:
        raise argparse.ArgumentTypeError("Model paramaters must be x,y,z")


def get_metric(predictions, targets):
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    aae = np.mean(np.abs(predictions - targets))
    r,_ = pearsonr(predictions,targets)
    return rmse, aae, r