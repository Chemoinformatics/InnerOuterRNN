from __future__ import print_function

import csv
import numpy as np
import itertools as it
import os
from mol_graph import graph_from_smiles_tuple, degrees




def read_csv(filename, nrows, input_name, target_name, delimiter = ','):
    data = ([], [])
    with open(filename) as file:
        reader = csv.DictReader(file, delimiter=delimiter)
#        if nrows is None:
#            nrows = 1000000
        try:
            for row in reader: #it.islice(reader, nrows):
#                print (row)
                val = float(row[target_name])
                data[0].append(row[input_name])
                data[1].append(val)
        except:
            print('Invalid label <{}> for input <{}>'.format(row[target_name], row[input_name]))
            pass
    return map(np.array, data)

    
def read_labels(filename, target_name):
    labels = []
    lent = len(target_name)
    with open(filename) as file:
        check_next = 0
        for line in file:
            if check_next:
                labels.append(line)
                check_next = 0
            elif len(line)>=lent and target_name in line:
                check_next = 1
    return labels



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





def load_huuskonsen(file = 'data/huuskonsen.csv', target_name = 'solubility'):
    '''
    returns: data, labels
    '''
    _alldata = read_csv(file, 1298, input_name='smiles', target_name=target_name)
    assert len(_alldata[0])==len(_alldata[1])
    data, labels = permute_data(_alldata[0], _alldata[1], FixSeed=12345)
    assert len(data)==len(labels)
    return data, labels
    
    


def load_delaney(file = 'data/delaney.csv', target_name = 'measured log solubility in mols per litre'):
    '''
    returns: data, labels
    '''
    _alldata = read_csv(file, 1128, input_name='smiles', target_name=target_name)
    assert len(_alldata[0])==len(_alldata[1])
    data, labels = permute_data(_alldata[0], _alldata[1], FixSeed=12345)
    assert len(data)==len(labels)
    return data, labels
    
    
def load_Karthikeyan_MeltingPoints(file = 'data/Melting_Points_(Karthikeyan).txt', target_name='MTP'):
    '''
    returns: data, labels
    '''
    _alldata = read_csv(file, 4451, input_name='SMILES', target_name=target_name)
    assert len(_alldata[0])==len(_alldata[1])
    data, labels = permute_data(_alldata[0], _alldata[1], FixSeed=12345)
    assert len(data)==len(labels)
    return data, labels
    

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
    
    
    
def load_SDF_file_wrapper(file = 'data/TD50__Mouse-Rat-Carconogenicity.sdf', target_name='TD50_Rat', lenient_target_format_enforcing = True):
    '''
    file:
    
        sdf file with smiles and other information like the targets
        
    lenient_target_format_enforcing:

        accept mixed float/string targets and try to extract the number
    
    returns: data, labels
    '''
    
    def wrapped_load(file_ = file, target_name_ = target_name, lenient_target_format_enforcing = lenient_target_format_enforcing):
        import rdkit
        suppl = rdkit.Chem.SDMolSupplier(file_)
        
        labels_raw = read_labels(file_, target_name=target_name_)
        
        if (len(labels_raw))==0:
            raise ValueError('ERROR:load_SDF_file_wrapper:: no entries found in SDF file with target_name =',target_name)
            
        
        invalid_labels_count = 0
        smiles = []
        labels = []
        for i, mol in enumerate(suppl):
            try:
                smile = rdkit.Chem.MolToSmiles(mol)
            except:
                continue
            
            try:
                l = labels_raw[i].replace('\n','')
                if lenient_target_format_enforcing:
                    try:
                        l = float(l)
                    except:
                        ok=0
                        for sp in l.split():
                            try:
                                l = float(sp)
                                ok=1
                                break
                            except:
                                pass
                        assert ok
                else:
                    l = float(l)
                smiles.append(smile)
                labels.append(l) # len(labels) <= len(labels_raw) as the smile extraction could fail for some molecules
            except:
                print('rejecting:',smile,'with invalid label:',l)
                invalid_labels_count += 1
                pass
        
        assert len(labels_raw)>=len(smiles), 'code probably crashed already if this happens'
        assert len(labels)==len(smiles), 'something went wrong'
        
        print('Total number of (valid) samples loaded:', len(smiles), 'out of originally ', len(labels_raw), 'rejecting',invalid_labels_count,'due to invalid or missing target value')
        smiles, labels = permute_data(smiles, labels, FixSeed=12345)
        return smiles, labels
    return wrapped_load
    
    
    
    
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





def array_rep_from_smiles(smiles):
    """extract features from molgraph"""
    molgraph = graph_from_smiles_tuple(smiles)
    arrayrep = {'atom_features' : molgraph.feature_array('atom'),
                'bond_features' : molgraph.feature_array('bond'),
                'atom_list'     : molgraph.neighbor_list('molecule', 'atom'), # List of lists.
                'rdkit_ix'      : molgraph.rdkit_ix_array()}  # For plotting only.
    for degree in degrees:
        arrayrep[('atom_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'atom'), dtype=int)
        arrayrep[('bond_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'bond'), dtype=int)
    return arrayrep


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


def filter_data(data_loading_function, data_cache_name = 'default_data_cache/'):
    """
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
    """
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
        


