'''
Blood-brain-barrier penetration classification task (Data set from: Hu Li, Chun Wei Yap, Choong Yong Ung, Ying Xue, Zhi Wei Cao and Yu Zong Chen, J. Chem. Inf. Model. 2005)
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #disables GPU detection, as multithreaded BLAS on CPU is faster in most cases; remove this line to enable the use of GPUs
import sys; sys.path.append('..') #makes this script runnable from the /examples subdir without adding adding /Outer to the pythonpath

import OuterModel.train_helper as train_helper
import OuterModel.train_outer as train_outer

target_field_name = 'binary_penetration'
input_data_csv = '../data/bbbp2__blood_brain_barrier_penetration.csv'
data_name='BBBP'


data, labels, regression, num_classes = train_helper.load_and_cache_csv(csv_file_name = input_data_csv,
                                                                 input_field_name  = 'smiles',
                                                                 target_field_name = target_field_name,
                                                                 cache_location = '../data/cached/')
xx_num_classes = 2



HyperParams = {'fp_depth': 3, 'conv_width': 240, 'fp_length': 247, 'predictor_MLP_layers': [472], 'batch_normalization': False, 'initial_lr': 0.0016549128,
               'num_MLP_layers': 1, 'training__num_epochs': 136, 'L2_reg': 0.00065333006}




model, train_scores, val_scores, test_scores = train_outer.perform_cross_validation(data, labels, HyperParams,
                                                                                    regression=False, num_classes=xx_num_classes,
                                                                                    initial_lr = HyperParams['initial_lr'],
                                                                                    L2_reg = HyperParams['L2_reg'],
                                                                                    use_matrix_based_implementation=False,
                                                                                    plot_training_mse=False, binary_multitask=False,
                                                                                    training__num_epochs = HyperParams['training__num_epochs'],
                                                                                    initial_crossval_index = 0,
                                                                                    crossval_total_num_splits = 10)



txt = ' Training:\n{}\n Validation:\n{}\n Testing:\n{}\n'.format('\n'.join(map(str, train_scores)),'\n'.join(map(str, val_scores)), '\n'.join(map(str, test_scores)))
train_helper.utils.save_text('results/{}_crossval_results.txt'.format(data_name), txt)



