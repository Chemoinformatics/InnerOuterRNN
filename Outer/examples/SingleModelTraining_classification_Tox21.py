'''
Tox21

this data set contains 12 different binary prediction targets
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #disables GPU detection, as multithreaded BLAS on CPU is faster in most cases; remove this line to enable the use of GPUs
import sys; sys.path.append('..') #makes this script runnable from the /examples subdir without adding adding /Outer to the pythonpath

import OuterModel.utils as utils
import OuterModel.data_preprocessing as data_preprocessing
import OuterModel.fingerprint_model_index_based as fingerprint_model

import OuterModel.train_helper as train_helper
import OuterModel.train_outer as train_outer


#select data set (csv-file) and the columns that are used as model input and prediction target:

target_field_names = 'NR-AR,NR-AR-LBD,NR-AhR,NR-Aromatase,NR-ER,NR-ER-LBD,NR-PPAR-gamma,SR-ARE,SR-ATAD5,SR-HSE,SR-MMP,SR-p53'.split(',')
data, labels, regression, num_classes = train_helper.load_and_cache_csv(csv_file_name = '../data/tox21.csv', input_field_name  = 'smiles',
                                                                       target_field_name = target_field_names, cache_location = '../data/cached/')
num_classes = labels.shape[1] if labels.ndim>1 else 1


# If labels contain missing entries (NaN) it is necessary to treat them specially.
# The following function cheks for NaNs and masks them if it does find them.
labels, masked_loss_function = train_helper.create_labels_NaN_mask(labels)


# construct the outer model. This is a "standard" Keras model with the associated methods like model.save_weights('filename') and model.load_weights('filename')
# 12 binary prediction targets, thus set: binary_multitask=True and regression=False
model = fingerprint_model.build_fingerprint_model(fp_length = 150, fp_depth = 3,
                                                    conv_width = 150, predictor_MLP_layers = [300, 300],
                                                    L2_reg = 1e-4, batch_normalization = False,
                                                    regression=False, number_of_classes = num_classes,
                                                    binary_multitask = True,
                                                    masked_loss_function = masked_loss_function)


# rudimentary data-set splitting:
split_proportions = [0.8, 0.1, 0.1] # train / validation / test
train_data, valid_data, test_data = utils.split_data_set(data, labels, split_proportions)


# transforms the SMILES into a list of dictionaries that encode features of the molecular graphs (uses RDKit).
train_data, valid_data, test_data = data_preprocessing.preprocess_data_set_for_Model(train_data, valid_data, test_data,
                                                                                     training_batchsize = 20,
                                                                                     testset_batchsize = 1000)

rval = train_outer.train_model(model, train_data, valid_data, test_data, initial_lr=0.002, total_lr_decay=0.01,
                                batchsize = 20, num_epochs = 120, regression = False)

# the training function returns the model and train/validation/test scores of the model at the point where it reached the best validation score (this is called "early stopping")
model, (train_scores_at_valbest, val_scores_best, test_scores_at_valbest), train_valid_mse_per_epoch, test_predictions = rval


train_helper.save_model_weights(model, 'results/Tox21_model_weights.npy')
# the weights can be loaded using train_helper.load_model_weights(model, 'results/Tox21_model_weights.npy')
# predictions can be made using train_helper.predict(model, _data_) if _data_ is a list of processed data dicts or directly via Keras' model.predict(_data_) if _data_ is a single dict

txt = '<Training set>:\n{}\n<Validation set>:\n{}\n<Test set>:\n{}\n'.format(train_helper.dict_as_txt(train_scores_at_valbest), train_helper.dict_as_txt(val_scores_best), train_helper.dict_as_txt(test_scores_at_valbest))
train_helper.utils.save_text('results/Tox21_results.txt', txt)
