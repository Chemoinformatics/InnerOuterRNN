'''
ESOL data set, also known under the name "Delaney"
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
data, labels, regression, num_classes = train_helper.load_and_cache_csv(csv_file_name = '../data/delaney.csv', input_field_name  = 'smiles',
                                                                        target_field_name = 'solubility', cache_location = '../data/cached/')
num_classes = labels.shape[1] if labels.ndim>1 else 1

# shifts the range of regression targets to the range of [0,1] to improve the convergence of trained neural network models.
labels, undo_normalization_fn = train_helper.normalize_RegressionTargets(labels)


# construct the outer model. This is a "standard" Keras model with the associated methods like model.save_weights('filename') and model.load_weights('filename')
model = fingerprint_model.build_fingerprint_model(fp_length = 84, fp_depth = 4,
                                                  conv_width = 106, predictor_MLP_layers = [359, 359],
                                                  L2_reg = 3e-5, batch_normalization = False,
                                                  regression=True, number_of_classes = num_classes,
                                                  binary_multitask = False,
                                                  masked_loss_function = False)


# rudimentary data-set splitting:
split_proportions = [0.8, 0.1, 0.1] # train / validation / test
train_data, valid_data, test_data = utils.split_data_set(data, labels, split_proportions)

train_data, valid_data, test_data = data_preprocessing.preprocess_data_set_for_Model(train_data, valid_data, test_data,
                                                                                     training_batchsize = 20,
                                                                                     testset_batchsize = 1000)

rval = train_outer.train_model(model, train_data, valid_data, test_data, initial_lr=0.04, total_lr_decay=0.01,
                                batchsize = 20, num_epochs = 5, regression = True,
                                Targets_UnNormalization_fn = undo_normalization_fn)
# the training function returns the model and train/validation/test scores of the model at the point where it reached the best validation score (this is called "early stopping")
model, (train_scores_at_valbest, val_scores_best, test_scores_at_valbest), train_valid_mse_per_epoch, test_predictions = rval

train_helper.save_model_weights(model, 'results/ESOL_model_weights.npy')
# the weights can be loaded using train_helper.load_model_weights(model, 'results/ESOL_model_weights.npy')
# predictions can be made using train_helper.predict(model, _data_) if _data_ is a list of processed data dicts or directly via Keras' model.predict(_data_) if _data_ is a single dict


if False:
    #this saves an image of the network's computational graph (an abstract form of it)
    #beware that this requires the 'graphviz' software!
    train_helper.save_model_visualization(model, filename = 'ESOL_model_visualization.png')
    train_helper.pyplot.show()

txt = '<Training set>:\n{}\n<Validation set>:\n{}\n<Test set>:\n{}\n'.format(train_helper.dict_as_txt(train_scores_at_valbest), train_helper.dict_as_txt(val_scores_best), train_helper.dict_as_txt(test_scores_at_valbest))
train_helper.utils.save_text('results/ESOL_results.txt', txt)
