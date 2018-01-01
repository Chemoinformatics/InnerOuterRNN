from __future__ import print_function
import csv
import numpy as np
import argparse
import os
from os import makedirs as _makedirs
from os.path import exists as _exists

from scipy.stats import pearsonr
import sklearn.metrics as metrics

#from . import input_data



def permute_data__simple(data):
    data_len = len(data)
    perm = np.random.permutation(data_len)
    data_perm = data[perm]
    return data_perm


def permute_data(data, labels=None, FixSeed=None, return_permutation=False, permutation = None):
    """Returns:
    data, labels (if both given) otherwise just data   , permutation [iff return_permutation==True]"""
    if FixSeed!=None:
        np.random.seed(int(FixSeed))
    s = np.shape(data)[0]
    if permutation is None:
        per = np.random.permutation(np.arange(s))
    else:
        per = permutation
    if type(data)==type([]):
        cpy = [data[i] for i in per]
    else:
        cpy = data[per]    #creates a copy! (fancy indexing)
    if labels is not None:
        if type(labels)==type([]):
            cpyl = [labels[i] for i in per]
        else:
            cpyl = labels[per]
        if not return_permutation:
            return cpy, cpyl
        else:
            return cpy, cpyl, per
    if not return_permutation:
        return cpy
    else:
        return cpy,per


def create_labels_NaN_mask(labels, force_masked = False):
    '''
    replaces NaN's with 0 in labels as well as attaching a channel containing the mask (1==keep, 0==discard).

    labels -> [labels, binary_weights ( = mask)]
    '''

    mask = np.where(np.isnan(labels), 0, 1).astype('int16')
    if not force_masked:
        if not np.any(mask==0):
            return labels, False
        print('Missing entries in labels detected: masked loss will be used. (masked labels shape: {})'.format(labels.shape))
    labels = np.nan_to_num(labels)
    labels = labels.astype('int16')
    labels = np.concatenate([labels[:,None,:],mask[:,None,:]], axis=1)

    return labels, True


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





def read_csv__old(filename, smile_name, target_name, logp_name=None):
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
    data = np.asarray(data, dtype=dt)
    return list(zip(*data))


def read_csv(filename, nrows, input_name, target_name, delimiter = ',', logp_col_name = None):
    if logp_col_name is not None:
        raise NotImplementedError('logp_col_name not supported as of now')
    data = []
    labels=[]
    with open(filename) as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        try:
            for row in reader:
                if isinstance(target_name, list):
                    val = np.asarray([(float(row[x]) if row[x]!='' else np.NAN) for x in target_name], 'float32')
                else:
                    val = float(row[target_name])
                data.append(row[input_name])
                labels.append(val)
        except:
            if isinstance(target_name, list):
                for x in target_name:
                    if not x in row:
                        print('Invalid label: {}'.format(x))
            else:
                print('Invalid label <{}> or <{}> for row <{}>'.format(input_name, target_name, row))
    return np.asarray(data), np.asarray(labels)#return map(np.asarray, data)


"""
def load_csv_file_wrapper(file = 'data/delaney.csv', input_name = 'smiles', target_name='solubility', delimiter=','):
    '''
    file:

        csv file with smiles and other information like the targets


    returns: data, labels
    '''

    def wrapped_load(file_ = file, input_name_ = input_name, target_name_ = target_name):
        _alldata = read_csv(file_, nrows=None, input_name=input_name_, target_name=target_name_, delimiter=delimiter)
        assert len(_alldata[0])==len(_alldata[1])
        assert len(_alldata[0])>0, 'nothing in CSV'
        print('smiles',_alldata[0])
        print('targets',_alldata[1])
        data, labels = permute_data(_alldata[0], _alldata[1], FixSeed=12345)
        assert len(data)==len(labels)
        return data, labels

    return wrapped_load





def filter_duplicates(data, labels):

    fast_dict = dict(zip(data, labels))
    if len(fast_dict)==len(data):
        return data, labels

    ret_dict = {}

    for d,l in zip(data, labels):
        if d in ret_dict:
            ret_dict[d] = (ret_dict[d][0] + 1, ret_dict[d][1] + l)
        else:
            ret_dict[d] = ( 1.,  l)

    print('Data set contained {} entries but {} were duplicates (unique elements: {}) -- averaging targets(labels) of duplicate entries'.format(len(data), len(data)-len(fast_dict) , len(fast_dict)))
    # keys/values will correctly match data/labels (but still shuffle it)
    return ret_dict.keys(), [summed/count for count, summed in ret_dict.values()]





def filter_data_inner(data_loading_function, data_cache_name = 'default_data_cache/'):
    '''
    loads data using <data_loading_function> (e.g. load_Karthikeyan_MeltingPoints()) and filters out all invalid SMILES.
    Saves the processed data on disk (name is specified by <data_cache_name>) and will re-load this file
    the next time filter_data() is called if the same <data_cache_name> is provided

    Inputs:
    ---------

        data_loading_function:

            a function returning two lists: a list of smiles(input data) and a list of labels/regression targets


        data_cache_name:

            string describing the location for storing the filtered data on disk.

            Set to None in order to disable this.
    '''
    try: #try to load cached files
        if data_cache_name is not None:
            data   = np.load(data_cache_name+'_data.npy')
            labels = np.load(data_cache_name+'_labels.npy')
        else:
            assert 0
    except:
        data_, labels_ = data_loading_function()# e.g. load_Karthikeyan_MeltingPoints()
        data_, labels_ = filter_duplicates(data_, labels_)
        data, labels = [ ],[]
        ok, banned = 0,0
        for i in range(len(data_)):
            try:
                mol = Molecule(data[i], molecule_logp, contract_rings)

                array_rep_from_smiles(data_[i:i+1])
                data.append(data_[i])
                labels.append(labels_[i])
                ok +=1
            except:
                banned +=1
        if data_cache_name is not None:
            print('removed', banned, 'and kept', ok,'samples')
        data = np.array(data)
        labels = np.array(labels)

        if data_cache_name is not None:
            try:
                os.makedirs('/'.join(data_cache_name.split('/')[:-1]))
            except:
                pass
            np.save(data_cache_name+'_data.npy', data)
            np.save(data_cache_name+'_labels.npy', labels)
    return data, labels





def load_and_cache_csv(csv_file_name = None, input_field_name = 'smiles', target_field_name = 'solubility'):
    '''
        Loads a csv-based data-set and caches the data afer processing.

        data_set (str or None) [optional]:
        --------------

            Specify the name of the cache and an example data set
            valid values: 'delaney', 'huuskonsen', 'bloodbrainbarrier', None

        Returns:
        ---------

            data, labels, regression, num_classes
    '''

    data_set = csv_file_name.replace('\\','/').split('/')[-1]

    loader_fn  = load_csv_file_wrapper(file = csv_file_name, input_name = input_field_name, target_name=target_field_name)

    # load and preprocess; uses cached data if available
    cache_name = 'data/cached/'+data_set
    data, labels = filter_data_inner(loader_fn, data_cache_name = cache_name)

    print('Number of valid examples in data set:',len(data))

    if isinstance(target_field_name, list):
        assert len(target_field_name) == labels.shape[1], 'ERROR: Failed to correctly load the data'

    return data, labels
"""


def save_results(file_path, targets, predictions, additional_str=''):
#    data = np.array([targets, predictions])
#    data = data.T
    if file_path[-1]!='/':
        file_path = file_path+'/'
    np.save(file_path+'targets'+additional_str,targets)
    np.save(file_path+'predictions'+additional_str,predictions)



def mkdir(path):
    if len(path)>1 and _exists(path)==0:
        _makedirs(path)

def extract_file_path(string):
    """  returns: path [will end in '/' unless empty]"""
    A = string.replace("\\","/").split("/")
    path = ("/".join(A[:-1]))+"/"
    if len(path)==1:
        path=""
    return path

def save_text(fname,string):
    path = extract_file_path(fname)
    if path!="" and os.path.exists(path)==False:
        mkdir(path)
    f=open(fname,'w')
    f.write(string)
    f.close()


def model_params_formatting(s):
    try:
        x, y, z = map(int, s.split(','))
        return x, y, z
    except:
        raise argparse.ArgumentTypeError("Model paramaters must be x,y,z")




def get_accuracy_AUC(predictions, labels, weights):
    auc = -1
    if predictions.ndim==2 and abs(np.mean(predictions.sum(1))-1) < 1e-4:
        #regular softmax predictions (binary)
        if labels.max()==1:
            predictions = predictions[:,1]
            fpr, tpr, thresholds = metrics.roc_curve(labels[:,1], predictions, pos_label=1)
            auc = metrics.auc(fpr, tpr, reorder=1)
        accuracy = np.mean( np.argmax(predictions, axis=1) == np.argmax(labels, axis=1) )
    else:
        # multi-task binary
        if weights is None:
            accuracy = np.mean(((np.squeeze(predictions)>0.5)==np.squeeze(labels)))
        else:
            assert weights.shape == labels.shape
            accuracy = np.sum(weights*((predictions>0.5)==labels).astype('int8'))*1./np.sum(weights)
        AUCs=[]

        if np.isnan(np.sum(predictions)):
            print('WARNING::get_accuracy_AUC: predictions contain NaN!')
            predictions = np.nan_to_num(predictions)
        for i in range(labels.shape[1]):
            fpr, tpr, thresholds = metrics.roc_curve(labels[:,i], predictions[:,i], pos_label=1, sample_weight = None if weights is None else weights[:,i] )
            try:
                auc = metrics.auc(fpr, tpr, reorder=1)
                if not np.isnan(auc):
                    AUCs.append(auc)
            except:
                pass

        auc = np.mean(AUCs)
        if np.isnan(auc):
            AUCs = np.asarray(AUCs)
            num_nan = np.sum(np.isnan(AUCs))
            auc = np.mean(np.nan_to_num(AUCs)) * len(AUCs)/(1.*len(AUCs)-num_nan) #mean ignoring NaN entries in AUCs
    return accuracy, auc


def get_metric(predictions, targets):
    #print('predictions, targets',predictions.shape, targets.shape)
    if targets.ndim==3 and targets.shape[1]==2:
        # weighted/masked binary classification
        labels, mask = targets[:,0,:], targets[:,1,:]
        accuracy, auc = get_accuracy_AUC(predictions, labels, mask)
        return {'accuracy':accuracy, 'auc':auc, 'primary':accuracy, 'secondary':auc}
    if targets.ndim==2 and targets.shape[1]==1: #predictions/targets (1732,) (1732, 1)
        # binary classification
        accuracy, auc = get_accuracy_AUC(predictions[:,None], targets, None)
        return {'accuracy':accuracy, 'auc':auc, 'primary':accuracy, 'secondary':auc}
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    aae = np.mean(np.abs(predictions - targets))

    r,_ = pearsonr(predictions, targets)
    r2  = metrics.r2_score(targets, predictions)
    return {'rmse':rmse, 'aae':aae, 'r':r, 'r2':r2, 'primary':rmse, 'secondary':r2}




def remove_SMILES_longer_than(data, max_length = 100):
    da = list(data[0])
    la = list(data[1])
    which = np.where(np.asarray(list(map(len, data[0])))>max_length)[0][::-1]
    removed_lengths = []
    for i in which:
        removed_lengths.append(len(da[i]))
        del da[i]
        del la[i]
    assert len(da)==len(la)
    if len(which):
        print('Removed {} SMILES that were longer than {} from data set ({} remaining) - consider increasing the max sequence length in the config file. [Longest smiles string had {} elements]'.format(len(which), max_length, len(la), max(removed_lengths)))
    return np.asarray(da), np.asarray(la)




