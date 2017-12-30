import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #disables GPU detection, as multithreaded BLAS on CPU is faster in most cases; remove this line to enable the use of GPUs
import sys; sys.path.append('..') #makes this script runnable from the /examples subdir without adding adding /Outer to the pythonpath

import OuterModel.train_helper as train_helper
import OuterModel.train_outer as train_outer

target_field_name = 'NR-AR,NR-AR-LBD,NR-AhR,NR-Aromatase,NR-ER,NR-ER-LBD,NR-PPAR-gamma,SR-ARE,SR-ATAD5,SR-HSE,SR-MMP,SR-p53'.split(',')
input_data_csv = '../data/tox21.csv'
data_name='Tox21'


data, labels, regression, num_classes = train_helper.load_and_cache_csv(csv_file_name = input_data_csv,
                                                                 input_field_name  = 'smiles',
                                                                 target_field_name = target_field_name,
                                                                 cache_location = '../data/cached/')
xx_num_classes = labels.shape[1] if labels.ndim>1 else 1




HyperParams = {'fp_length':150, 'conv_width':150, 'fp_depth':3,
               'predictor_MLP_layers':[300, 300], 'L2_reg': 1e-4, 'initial_lr': 0.002,
               'training__num_epochs':100, 'batch_normalization':False}



model, train_scores, val_scores, test_scores = train_outer.perform_cross_validation(data, labels, HyperParams,
                                                                                    regression=False, num_classes=xx_num_classes,
                                                                                    initial_lr = HyperParams['initial_lr'],
                                                                                    L2_reg = HyperParams['L2_reg'],
                                                                                    use_matrix_based_implementation=False,
                                                                                    plot_training_mse=False, binary_multitask=True,
                                                                                    training__num_epochs = HyperParams['training__num_epochs'],
                                                                                    initial_crossval_index = 0,
                                                                                    crossval_total_num_splits = 10)



txt = '<Training set>:\n{}\n\n<Validation set>:\n{}\n\n<Test set>:\n{}\n'.format('\n'.join(map(str, train_scores)),'\n'.join(map(str, val_scores)), '\n'.join(map(str, test_scores)))
train_helper.utils.save_text('results/{}_crossval_results.txt'.format(data_name), txt)



