'''
Blood-brain-barrier penetration classification task
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '' #disables GPU detection, as multithreaded BLAS on CPU is faster in most cases; remove this line to enable the use of GPUs
import sys; sys.path.append('..') #makes this script runnable from the /examples subdir without adding adding /Outer to the pythonpath

import InnerModel.train as train_inner


target_col = 'p_np'
input_data_csv = '../data/BBBP.csv'
data_name = 'BBBP'




HyperParams = {'mparam3': 65, 'mparam2': 58, 'mparam1': 6, 'lr': 0.002, 'batch_size': 10, 'train_n_epochs':100}
Output_dir='{}_model_output/'.format(data_name)


training_scores_dict, validation_scores_dict = train_inner.main(output_dir=Output_dir,
                                                                model_name='my_{}_model'.format(data_name), training_file=input_data_csv,
                                                                validation_file=None, smile_col='smiles', target_col=target_col,
                                                                crossval_total_num_splits=10, experiment_name=data_name,
                                                                regression=False, binary_classification=True,
                                                                batch_size = HyperParams['batch_size'], clip_gradient=False,
                                                                model_params = [HyperParams['mparam1'], HyperParams['mparam2'], HyperParams['mparam3']],
                                                                contract_rings=False, learning_rate = HyperParams['lr'],
                                                                max_epochs=HyperParams['train_n_epochs'], enable_plotting=False)

text = '<Training set scores>:\n{}\n\n<Validation set scores>:\n{}'.format('\n'.join(map(str,training_scores_dict)), '\n'.join(map(str,validation_scores_dict)))
train_inner.utils.save_text('{}{}_crossvalidation.txt'.format(Output_dir, data_name), text)
