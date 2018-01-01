import os
os.environ['CUDA_VISIBLE_DEVICES'] = '' #disables GPU detection, as multithreaded BLAS on CPU is faster in most cases; remove this line to enable the use of GPUs
import sys; sys.path.append('..') #makes this script runnable from the /examples subdir without adding adding /Outer to the pythonpath

import InnerModel.train as train_inner

target_col = 'NR-AR,NR-AR-LBD,NR-AhR,NR-Aromatase,NR-ER,NR-ER-LBD,NR-PPAR-gamma,SR-ARE,SR-ATAD5,SR-HSE,SR-MMP,SR-p53'.split(',')


input_data_csv = '../data/tox21.csv'
data_name = 'tox21'

Output_dir = '{}_model_output/'.format(data_name)


HyperParams = {'mparam3': 34, 'mparam2': 45, 'mparam1': 7, 'batch_size': 9, 'lr': 0.00052664289, 'contract_rings': 0}


training_scores_dict, validation_scores_dict = train_inner.main(output_dir=Output_dir,
                model_name='my_tox21_model_1', logp_col='', add_logp=False, training_file=input_data_csv,
                 validation_file=None, smile_col='smiles', target_col=target_col,
                 crossval_total_num_splits=10, initial_crossvalidation_index=0, experiment_name='tox21',
                 regression=0, binary_classification=1, batch_size = HyperParams['batch_size'], clip_gradient=0,
                 model_params = [HyperParams['mparam1'], HyperParams['mparam2'], HyperParams['mparam3']], contract_rings=0, learning_rate = HyperParams['lr'],
                 max_epochs=100, enable_plotting=0)

text = '<Training set scores>:\n{}\n\n<Validation set scores>:\n{}'.format('\n'.join(map(str,training_scores_dict)), '\n'.join(map(str,validation_scores_dict)))
train_inner.utils.save_text('{}{}_crossvalidation.txt'.format(Output_dir, data_name), text)
