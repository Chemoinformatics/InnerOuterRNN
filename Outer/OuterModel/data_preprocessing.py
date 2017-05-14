import utils, config
import numpy as np


def connectivity_to_Matrix_list(list_of_neighbors_lists, total_num_features):
    """
    not used
    """
    assert isinstance(list_of_neighbors_lists,list)
    offsets = [x.shape[0] for x in list_of_neighbors_lists]
    N = sum(offsets)
    mat = np.zeros((N, total_num_features),'float32')
    offset=0
    for off, neigh_list in zip(offsets, list_of_neighbors_lists):
        for i,x in enumerate(neigh_list):
            mat[i+offset,x] = 1
        offset += off
    return mat




def connectivity_to_Matrix(neighbors_lists, total_num_features):
    """
    Returns a matrix with binary entries that stores atom-neighborhood information of the molecule. 
    Multiplying a vector/matrix of atom features with this matrix will return the summation of all neigboring atom featues.
    
    Only used for atom features, as bond-features are not updated by the DNN layers and are thus "constant"
    """
    N = len(neighbors_lists)
    mat = np.zeros((N, total_num_features),'float32')    
    for i,x in enumerate(neighbors_lists):
        mat[i,x] = 1        
    return mat





def extract_bondfeatures_of_neighbors_by_degree(array_rep):
    """
    Sums up all bond features that connect to the atoms (sorted by degree)
    
    Returns:
    ----------
    
    list with elements of shape: [(num_atoms_degree_0, 6), (num_atoms_degree_1, 6), (num_atoms_degree_2, 6), etc....]
    
    e.g.:
    
    >> print [x.shape for x in extract_bondfeatures_of_neighbors_by_degree(array_rep)]
    
    [(0,), (269, 6), (524, 6), (297, 6), (25, 6), (0,)]  
    
    """
    bond_features_by_atom_by_degree = []
    for degree in config.ATOM_DEGREES:
        bond_features = array_rep['bond_features']
        bond_neighbors_list = array_rep[('bond_neighbors', degree)]
        summed_bond_neighbors = bond_features[bond_neighbors_list].sum(axis=1)
        bond_features_by_atom_by_degree.append(summed_bond_neighbors)
    return bond_features_by_atom_by_degree



def _validate_data(batches):
    
    for batch in batches:
        SUM=0
        for x in batch[0].keys():
            if 'atom_features_selector_matrix_degree_' in x:
                SUM += batch[0][x].shape[0]
        assert SUM == batch[0]['atom_features_selector_matrix_degree_1'].shape[1], '\n\nInconsistency in data detected! Model will fail to run.\nConsider changing entries in <config.py>!'
        
        




def _preprocess_data(smiles, labels, batchsize = 100):
    """
    prepares all input batches to train/test the GDNN fingerprints implementation
    """
    N = len(smiles)
    batches = []
    
    num_bond_features = 6
    
    for i in range(int(np.ceil(N*1./batchsize))):
        array_rep = utils.array_rep_from_smiles(smiles[i*batchsize:min(N,(i+1)*batchsize)])
        labels_b = labels[i*batchsize:min(N,(i+1)*batchsize)]
        atom_features = array_rep['atom_features']

        summed_bond_features_by_degree = extract_bondfeatures_of_neighbors_by_degree(array_rep)

        batch_dict = {'input_atom_features':atom_features} # (num_atoms, num_atom_features)

        missing_degrees = []
        for degree in config.ATOM_DEGREES:
        
            atom_neighbors_list = array_rep[('atom_neighbors', degree)]
            if len(atom_neighbors_list)==0:
                missing_degrees.append(degree)
                continue
            
            # this matrix is used by every layer to match and sum all neighboring updated atom features to the atoms
            atom_neighbor_matching_matrix = connectivity_to_Matrix(atom_neighbors_list, atom_features.shape[0])
            atom_batch_matching_matrix = connectivity_to_Matrix(array_rep['atom_list'], atom_features.shape[0]).T

            assert np.all(atom_batch_matching_matrix.sum(1).mean()==1)
            assert np.all(atom_batch_matching_matrix.sum(0).mean()>1),'Error: looks like a single-atom molecule?'

            
            batch_dict['bond_features_degree_'+str(degree)] = summed_bond_features_by_degree[degree]

            batch_dict['atom_neighbors_indices_degree_'+str(degree)] = atom_neighbors_list
            batch_dict['atom_features_selector_matrix_degree_'+str(degree)] = atom_neighbor_matching_matrix
            batch_dict['atom_batch_matching_matrix_degree_'+str(degree)] = atom_batch_matching_matrix.T # (batchsize, num_atoms)
            
#            if degree==0:
#                print 'degree 0 bond?'
#                print smiles[i*batchsize:min(N,(i+1)*batchsize)]
#                return
                
#            input_atom_features (292L, 62L)
#            bond_features_degree_ 1 (70L, 6L)
#            atom_neighbors_indices_degree_ 1 (70L, 1L)
#            bond_features_degree_ 2 (134L, 6L)
#            atom_neighbors_indices_degree_ 2 (134L, 2L)
#            bond_features_degree_ 3 (78L, 6L)
#            atom_neighbors_indices_degree_ 3 (78L, 3L)
#            bond_features_degree_ 4 (10L, 6L)
#            atom_neighbors_indices_degree_ 4 (10L, 4L)
            num_bond_features = batch_dict['bond_features_degree_'+str(degree)].shape[1]
            num_atoms = atom_neighbor_matching_matrix.shape[1]
        for missing_degree in missing_degrees:
            batch_dict['atom_neighbors_indices_degree_'+str(missing_degree)] = np.zeros((0, missing_degree),'int32')
            batch_dict['bond_features_degree_'+str(missing_degree)] = np.zeros((0, num_bond_features),'float32')
            batch_dict['atom_features_selector_matrix_degree_'+str(missing_degree)] = np.zeros((0, num_atoms),'float32') 
            batch_dict['atom_batch_matching_matrix_degree_'+str(missing_degree)] = atom_batch_matching_matrix.T
        batches.append((batch_dict,labels_b))
    _validate_data(batches)
    return batches





def preprocess_data_set_for_Model(traindata, valdata, testdata, training_batchsize = 50, testset_batchsize = 1000):
    
    train = _preprocess_data(traindata[0], traindata[1], training_batchsize)
    validation = _preprocess_data(valdata[0],  valdata[1],  testset_batchsize )
    test = _preprocess_data(testdata[0], testdata[1], testset_batchsize )
    
    return train, validation, test





















if __name__=='__main__':
    
    data, labels = utils.load_delaney()
    traindata, valdata, testdata = utils.cross_validation_split(data, labels, crossval_split_index=0, 
                                                                crossval_total_num_splits=10, 
                                                                validation_data_ratio=0.1)
    
    preprocess_data_set(traindata, valdata, testdata, training_batchsize = 50, test_batchsize = 1000)



    def test__main(array_rep):

        r = extract_bondfeatures_of_neighbors_by_degree(array_rep)
        print r
        atom_features = array_rep['atom_features']
        bond_features = array_rep['bond_features']
        print 'atom_features',atom_features.shape
        print ' '*38,'bond_features',bond_features.shape
        for i in range(0,5):
            atom_neighbors_list = array_rep[('atom_neighbors', i)]
            bond_neighbors_list = array_rep[('bond_neighbors', i)]
            if len(atom_neighbors_list)==0:
                continue
            print 'degree',i
            print 'atom_neighbors_list',atom_neighbors_list.min(), atom_neighbors_list.max(), atom_neighbors_list.shape
            print ' '*38,'bond_neighbors_list',bond_neighbors_list.min(), bond_neighbors_list.max(), bond_neighbors_list.shape
        


    def test__connectivity_to_Matrix():
        print
        print '*'*30
        print
        pairs = np.array([[0, 2], [0, 3], [2, 3], [1, 3], [0, 2], [1, 2], [1, 0], [0 ,3], [0 ,2]])
        triplets = np.array([[3 ,1, 0], [2 ,3, 0], [0, 2, 1], [3 ,2 ,1], [1 ,3, 0], [1 ,2, 3]])
        print pairs
        print triplets
        print connectivity_to_Matrix(pairs, 4), connectivity_to_Matrix(pairs, 4).shape
        print connectivity_to_Matrix(triplets, 4)    , connectivity_to_Matrix(triplets, 4).shape
        print connectivity_to_Matrix_list([pairs,triplets], 4), connectivity_to_Matrix_list([pairs,triplets], 4).shape
        print
        print '*'*30
        print



    def test():
        (traindata_SMILES, trainlabels), valdata, testdata = utils.load_delaney()
        array_rep = utils.array_rep_from_smiles(traindata_SMILES[:2000])
        for degree in range(6):
            atom_neighbors_list = array_rep[('atom_neighbors', degree)]
            print 'degree',degree
            if len(atom_neighbors_list):
                print atom_neighbors_list.shape, atom_neighbors_list.min(), atom_neighbors_list.max()
        test__main(array_rep)
        
    test()

    test__connectivity_to_Matrix()




















