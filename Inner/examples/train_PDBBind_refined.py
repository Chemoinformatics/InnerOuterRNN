import os
os.environ['CUDA_VISIBLE_DEVICES'] = '' #disables GPU detection, as multithreaded BLAS on CPU is faster in most cases; remove this line to enable the use of GPUs
import sys; sys.path.append('..') #makes this script runnable from the /examples subdir without adding adding /Outer to the pythonpath

import InnerModel.train as train_inner

input_data_csv = '../data/PDBBind_refined_smiles_labels.csv'
target_col = '-logKd/Ki'
data_name = 'PDBBind_refined'

Output_dir = '{}_model_output/'.format(data_name)

HyperParams = {'mparam3': 56, 'mparam2': 44, 'mparam1': 9, 'batch_size': 9, 'lr': 0.0013787709, 'MAX_epochs': 82, 'weight_decay_factor': 0.00010428458}


training_scores_dict, validation_scores_dict = train_inner.main(output_dir=Output_dir,
                model_name='my_{}_model'.format(data_name), logp_col='', add_logp=False, training_file=input_data_csv,
                 validation_file=None, smile_col='smiles', target_col=target_col,
                 crossval_total_num_splits=10, experiment_name='{}'.format(data_name),
                 regression=1, binary_classification=0, batch_size = HyperParams['batch_size'], clip_gradient=0,
                 model_params = [HyperParams['mparam1'], HyperParams['mparam2'], HyperParams['mparam3']], contract_rings=0,
                 learning_rate = HyperParams['lr'], max_epochs=HyperParams['MAX_epochs'], enable_plotting=0,
                 weight_decay_factor = HyperParams['weight_decay_factor'])

text = '<Training set scores>:\n{}\n\n<Validation set scores>:\n{}'.format('\n'.join(map(str,training_scores_dict)), '\n'.join(map(str,validation_scores_dict)))
train_inner.utils.save_text('{}{}_crossvalidation.txt'.format(Output_dir, data_name), text)

