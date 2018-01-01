import os
os.environ['CUDA_VISIBLE_DEVICES'] = '' #disables GPU detection, as multithreaded BLAS on CPU is faster in most cases; remove this line to enable the use of GPUs
import sys; sys.path.append('..') #makes this script runnable from the /examples subdir without adding adding /Outer to the pythonpath

import InnerModel.train as train_inner

input_data_csv = '../data/karthikeyan/melting_points.csv'
target_col = 'MTP'
data_name = 'karthikeyan'



Output_dir = '{}_model_output/'.format(data_name)

HyperParams = {'mparam3': 22, 'mparam2': 26, 'mparam1': 11, 'batch_size': 8, 'lr': 0.00045462663, 'MAX_epochs': 142, 'weight_decay_factor': 4.5505141e-05}



training_scores_dict, validation_scores_dict = train_inner.main(output_dir=Output_dir,
                model_name='my_{}_model'.format(data_name), logp_col='', add_logp=False, training_file=input_data_csv,
                 validation_file=None, smile_col='SMILES', target_col=target_col,
                 crossval_total_num_splits=10, experiment_name='{}'.format(data_name),
                 regression=1, binary_classification=0, batch_size = HyperParams['batch_size'], clip_gradient=0,
                 model_params = [HyperParams['mparam1'], HyperParams['mparam2'], HyperParams['mparam3']], contract_rings=0,
                 learning_rate = HyperParams['lr'], max_epochs=HyperParams['MAX_epochs'], enable_plotting=0,
                 weight_decay_factor = HyperParams['weight_decay_factor'])

text = '<Training set scores>:\n{}\n\n<Validation set scores>:\n{}'.format('\n'.join(map(str,training_scores_dict)), '\n'.join(map(str,validation_scores_dict)))
train_inner.utils.save_text('{}{}_crossvalidation.txt'.format(Output_dir, data_name), text)

