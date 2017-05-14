import numpy as np

from utils import read_csv, cross_validation_split, permute_data
from rdkit import Chem

def split_delaney():
    csv_file_path = 'ugrnn/data/delaney/delaney.csv'
    smile_col_name = "smiles"
    target_col_name = "solubility"
    logp_col_name = "logp"

    data = read_csv(csv_file_path, smile_col_name, target_col_name, logp_col_name)
    data_perm = permute_data(data)

    traindata, valdata, testdata = cross_validation_split(data_perm, crossval_split_index=1, crossval_total_num_splits=10)

    train_file_path = 'ugrnn/data/delaney/train_delaney.csv'
    validate_file_path = 'ugrnn/data/delaney/validate_delaney.csv'
    test_file_path = 'ugrnn/data/delaney/test_delaney.csv'

    header = "{:},{:},{:}".format(smile_col_name, target_col_name, logp_col_name )
    fmt = ('%s', '%4f', '%4f')
    np.savetxt(train_file_path, traindata, header=header, fmt=fmt, comments='', delimiter=',')
    np.savetxt(validate_file_path, valdata, header=header, fmt=fmt, comments='', delimiter=',')
    np.savetxt(test_file_path, testdata, header=header, fmt=fmt, comments='', delimiter=',')

def valid_smile(smile):
    return ('.' not in smile and Chem.MolFromSmiles(smile) is not None)

def split_karthikeyan():
    csv_file_path = 'ugrnn/data/karthikeyan/melting_points.csv'
    smile_col_name = "SMILES"
    target_col_name = "MTP"

    data = read_csv(csv_file_path, smile_col_name, target_col_name)
    bool_arr = np.array([valid_smile(row[0]) for row in data])
    print(bool_arr)
    filter_data = data[bool_arr]
    data_perm = permute_data(filter_data)

    traindata, valdata, testdata = cross_validation_split(data_perm, crossval_split_index=1, crossval_total_num_splits=10)

    train_file_path = 'ugrnn/data/karthikeyan/train_karthikeyan.csv'
    validate_file_path = 'ugrnn/data/karthikeyan/validate_karthikeyan.csv'
    test_file_path = 'ugrnn/data/karthikeyan/test_karthikeyan.csv'

    header = "{:},{:}".format(smile_col_name, target_col_name)
    fmt = ('%s', '%4f')
    np.savetxt(train_file_path, traindata, header=header, fmt=fmt, comments='', delimiter=',')
    np.savetxt(validate_file_path, valdata, header=header, fmt=fmt, comments='', delimiter=',')
    np.savetxt(test_file_path, testdata, header=header, fmt=fmt, comments='', delimiter=',')

def split_huuskonsen():
    train_file_path = 'ugrnn/data/huuskonsen/train.smi'
    test1_file_path = 'ugrnn/data/huuskonsen/test1.smi'
    test2_file_path = 'ugrnn/data/huuskonsen/test2.smi'

    smile_col_name = "smiles"
    target_col_name = "solubility"
    logp_col_name = "logp"

    dtype = [(smile_col_name, 'S200'), (target_col_name, 'f8'), (logp_col_name, 'f8')]

    data = np.genfromtxt(train_file_path, usecols=(6,3,5), dtype=dtype, comments=None)
    data_perm = permute_data(data)

    l = len(data)
    train_end = int(l*.9)

    train_data = data_perm[:train_end]
    val_data = data_perm[train_end:]

    test1_data = np.genfromtxt(test1_file_path, usecols=(6, 3, 5), dtype=dtype)
    test2_data = np.genfromtxt(test2_file_path, usecols=(6, 3, 5), dtype=dtype)
    test_data = np.concatenate((test1_data , test2_data))

    train_file_path = 'ugrnn/data/huuskonsen/train_huuskonsen.csv'
    validate_file_path = 'ugrnn/data/huuskonsen/validate_huuskonsen.csv'
    test_file_path = 'ugrnn/data/huuskonsen/test_huuskonsen.csv'

    header = "{:},{:},{:}".format(smile_col_name, target_col_name, logp_col_name)
    fmt = ('%s', '%4f', '%4f')
    np.savetxt(train_file_path, train_data, header=header, fmt=fmt, comments='', delimiter=',')
    np.savetxt(validate_file_path, val_data, header=header, fmt=fmt, comments='', delimiter=',')
    np.savetxt(test_file_path, test_data, header=header, fmt=fmt, comments='', delimiter=',')


#split_karthikeyan()