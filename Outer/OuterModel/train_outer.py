"""
Specifies, trains and evaluates (cross-validation) neural fingerprint model.

Performs early-stopping on the validation set.

The model is specified inside the main() function, which is a demonstration of this code-base

"""
from __future__ import print_function

import numpy as np

import warnings
import time

from . import utils
from . import data_preprocessing
from . import fingerprint_model_matrix_based
from . import fingerprint_model_index_based
from . import train_helper

NORMALIZE_REGRESSION_TARGETS_DURING_TRAINING = 1 #normalize to 0..1 range; this will be undone before evaluating model predictions


Default_CNN_model_hyperparamters = {'fp_length':51, 'conv_width':50, 'fp_depth':3,
                                    'predictor_MLP_layers':[100, 100, 100],
                                    'batch_normalization':False}



def lim(float, precision = 5):
    return ("{0:."+str(precision)+"f}").format(float)






def collect_apply(list_of_dicts, key, fn_to_apply = lambda x:x):
    rv = []
    for x in list_of_dicts:
        rv.append(x[key])
    return fn_to_apply(rv)


def nice_prnt(Type_printed, dict_key, train_scores, val_scores, test_scores, crossval_total_num_splits):
    ret  = 'Mean  training  {} = {:.5f} +- {:.5f}\n'.format(Type_printed, collect_apply(train_scores,dict_key, np.mean), collect_apply(train_scores,dict_key, np.std)/np.sqrt(crossval_total_num_splits))
    ret += 'Mean validation {} = {:.5f} +- {:.5f}\n'.format(Type_printed, collect_apply(val_scores,  dict_key, np.mean), collect_apply(val_scores,  dict_key, np.std)/np.sqrt(crossval_total_num_splits))
    ret += 'Mean  testing   {} = {:.5f} +- {:.5f}\n'.format(Type_printed, collect_apply(test_scores, dict_key, np.mean), collect_apply(test_scores, dict_key, np.std)/np.sqrt(crossval_total_num_splits))
    return ret



def perform_cross_validation(data, labels,
                             CNN_model_hyperparamters,
                             regression  = True,
                             num_classes = 1,
                             binary_multitask = False,
                             use_matrix_based_implementation = False,
                             plot_training_mse    = False,
                             training__num_epochs = 100,
                             training__batchsize  = 20,
                             initial_lr           = 3e-3,
                             L2_reg               = 1e-3,
                             batch_normalization  = 0,
                             initial_crossval_index = 0,
                             crossval_total_num_splits = 10):
    """
    Performs a full cross-validation training & testing experiment; trains a network <crossval_total_num_splits> many times.

    use_matrix_based_implementation:

        There are two different (but equivalent!) implementations of neural-fingerprints,
        which can be selected with the binary parameter <use_matrix_based_implementation>

    binary_multitask:

        set to true for e.g. Tox21 where num_classes=12 many binary classification tasks are to be run in parallel

    Returns:
    -----------
        Lists of dictionaries with metrics:

        model, train_scores, val_scores, test_scores
    """

    np.random.seed(1338)    # for reproducibility


    #~~~~~~~~~~~~~~~~~~~~~~~~~
    fp_length  = CNN_model_hyperparamters['fp_length']  # size of the final constructed fingerprint vector
    conv_width = CNN_model_hyperparamters['conv_width'] # number of filters per fingerprint layer
    fp_depth   = CNN_model_hyperparamters['fp_depth']   # number of convolutional fingerprint layers
    predictor_MLP_layers = CNN_model_hyperparamters['predictor_MLP_layers']
    #~~~~~~~~~~~~~~~~~~~~~~~~~



    # select the data that will be loaded or provide different data

    train_scores = []
    val_scores   = []
    test_scores  = []

    all_test_predictions = []
    all_test_labels      = [] #in original order of the data, but this can change when <cross_validation_split> changes

    if use_matrix_based_implementation:
        fn_build_model   = fingerprint_model_matrix_based.build_fingerprint_regression_model
    else:
        fn_build_model   = fingerprint_model_index_based.build_fingerprint_model

    if binary_multitask:
        assert num_classes==labels.shape[1],'USER INPUT ERROR: num_classes ({}) and labels in data ({}) are not matching'.format(num_classes, labels.shape[1])
        labels, masked_loss_function = train_helper.create_labels_NaN_mask(labels)
    else:
        masked_loss_function = 0

    if regression:
        print('Naive baseline (using mean): MSE =', lim(np.mean((labels-labels.mean())**2)), '(RMSE =', train_helper.lim(np.sqrt(np.mean((labels-labels.mean())**2))),')')

        if NORMALIZE_REGRESSION_TARGETS_DURING_TRAINING:
            labels, Targets_UnNormalization_fn = train_helper.normalize_RegressionTargets(labels)
        else:
            Targets_UnNormalization_fn = lambda x:x
    else:
        if not binary_multitask:
            labels = labels.astype('int16')
            print('Naive baseline (guessing): Accuracy = {}%'.format(lim(  100*max([np.mean((labels==v)) for v in range(num_classes)])   )))
            tmp = np.zeros((len(labels), num_classes),'int16')
            for i,l in enumerate(labels):
                tmp[i,l]=1
            labels = tmp
        Targets_UnNormalization_fn = lambda x:x


    for crossval_split_index in range(initial_crossval_index, crossval_total_num_splits):
        print('\ncrossvalidation split',crossval_split_index+1,'of',crossval_total_num_splits)

        print('splitting data...')
        traindata, valdata, testdata = utils.cross_validation_split(data, labels, crossval_split_index=crossval_split_index,
                                                                    crossval_total_num_splits=crossval_total_num_splits,
                                                                    validation_data_ratio=0.1)

        print('preprocessing data...')
        train, valid_data, test_data = data_preprocessing.preprocess_data_set_for_Model(traindata, valdata, testdata,
                                                                     training_batchsize = training__batchsize,
                                                                     testset_batchsize = 1000)
        print('building model...')


        model = fn_build_model(fp_length = fp_length, fp_depth = fp_depth,
                               conv_width = conv_width, predictor_MLP_layers = predictor_MLP_layers,
                               L2_reg = L2_reg, num_input_atom_features = 62,
                               num_bond_features = 6, batch_normalization = batch_normalization,
                               regression=regression, number_of_classes = num_classes,
                               binary_multitask = binary_multitask,
                               masked_loss_function = masked_loss_function)


        model, (train_scores_at_valbest, val_scores_best, test_scores_at_valbest), train_valid_mse_per_epoch, test_predictions = \
                         train_model(model, train, valid_data, test_data, initial_lr=initial_lr, total_lr_decay=0.01,
                                     batchsize = training__batchsize, num_epochs = training__num_epochs, train=1,
                                     regression = regression, Targets_UnNormalization_fn = Targets_UnNormalization_fn)
        train_scores.append(train_scores_at_valbest)
        val_scores.append(val_scores_best)
        test_scores.append(test_scores_at_valbest)

        all_test_predictions.append(test_predictions[:,0])
        all_test_labels.append(np.concatenate(list(map(lambda x:x[-1],test_data))))

        if plot_training_mse:
            train_helper.plot_training_mse_evolution(train_valid_mse_per_epoch, ['training set MSE (+regularizer)', 'validation set MSE/accuracy'])
            train_helper.pyplot.draw()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                train_helper.pyplot.pause(0.0001)


    print('\n\nCrossvalidation complete!\n')
    Type = 'MSE' if regression else 'accuracy'
    results_txt = nice_prnt(Type, Type.lower(), train_scores, val_scores, test_scores, crossval_total_num_splits)

    if regression:
        results_txt += nice_prnt('MAE(AAE)', 'mae', train_scores, val_scores, test_scores, crossval_total_num_splits)
        results_txt += nice_prnt('RMSE', 'rmse', train_scores, val_scores, test_scores, crossval_total_num_splits)
        results_txt += nice_prnt('R2', 'r2', train_scores, val_scores, test_scores, crossval_total_num_splits)

    elif 'auc' in train_scores[0]:
        results_txt += nice_prnt('AUC', 'auc', train_scores, val_scores, test_scores, crossval_total_num_splits)

    avg_test_scores = np.array([list(x.values()) for x in test_scores]).mean(0)
    avg_test_scores_dict = dict(zip(list(test_scores[0].keys()), avg_test_scores))

    print(results_txt)
    print()
    for k,v in avg_test_scores_dict.items():
        if k not in ['mse','rmse']:#filter duplicates
            print('Test-set',k,'=',lim(v))

    if regression:
        train_helper.parity_plot(np.concatenate(all_test_predictions), np.concatenate(all_test_labels))

    return model, train_scores, val_scores, test_scores










def train_model(model, train_data, valid_data, test_data,
                 batchsize = 20, num_epochs = 100, train = True,
                 initial_lr=3e-3, total_lr_decay=0.2, verbose = 1,
                 regression = True, Targets_UnNormalization_fn = lambda x:x):
    """
    Main training loop for the DNN.

    Input:
    ---------

    train_data, valid_data, test_data:

        lists of tuples (data-batch, labels-batch)

    total_lr_decay:

        value in (0, 1] -- this is the inverse total LR reduction factor over the course of training

    verbose:

        value in [0,1,2] -- 0 print minimal information (when training ends), 1 shows training loss, 2 shows training and validation loss after each epoch


    Returns:
    -----------

        model (keras model object) -- model/weights selected by early-stopping on the validation set (model at epoch with lowest validation error)

        3-tuple of train/validation/test dictionaries (as returned by eval_metrics_on(...) )

        2-tuple of training/validation-set MSE after each epoch of training

    """

    if train:

        log_train_loss = []
        log_validation_loss = []

        if verbose>0:
            print('starting training (compiling)...')

        best_valid = 9e9
        model_params_at_best_valid=[]

        times=[]
        for epoch in range(num_epochs):
            train_helper.update_lr(model, initial_lr, epoch*1./num_epochs, total_lr_decay)
            batch_order = np.random.permutation(len(train_data))
            losses=[]
            t0 = time.clock()
            for i in batch_order:
#                print (i,'train_data[i][1]',train_data[i][1].shape)
                loss = model.train_on_batch(x=train_data[i][0], y=train_data[i][1], check_batch_dim=False)
                losses.append(loss)
            times.append(time.clock()-t0)

            normalized_val_mse = train_helper.test_on(valid_data, model,'valid_data score:' if verbose>1 else '')
            if best_valid > normalized_val_mse:
                best_valid = normalized_val_mse
                model_params_at_best_valid = train_helper.get_model_params(model) #kept in RAM (not saved to disk as that is slower)
            if verbose>0:
                print('Epoch {}/{} completed with average loss {}'.format(epoch+1, num_epochs, lim(np.mean(losses))))
            log_train_loss.append(np.mean(losses))
            log_validation_loss.append(normalized_val_mse)

        # excludes times[0] as it includes compilation time
        print('Training @',lim(1./np.mean(times[1:])),'epochs/sec (',lim(batchsize*len(train_data)/np.mean(times[1:])),'examples/s)')



    train_helper.set_model_params(model, model_params_at_best_valid)

    training_data_scores   = train_helper.eval_metrics_on(train_helper.predict(train_data,model), train_data, regression = regression, Targets_UnNormalization_fn=Targets_UnNormalization_fn)
    validation_data_scores = train_helper.eval_metrics_on(train_helper.predict(valid_data,model), valid_data, regression = regression, Targets_UnNormalization_fn=Targets_UnNormalization_fn)
    test_predictions       = train_helper.predict(test_data,model)
    test_data_scores       = train_helper.eval_metrics_on(test_predictions, test_data, regression = regression, Targets_UnNormalization_fn=Targets_UnNormalization_fn)


    if regression:
        print('\ntraining   set mse (best_val):', lim(training_data_scores['mse']))
        print('validation set mse (best_val):', lim(validation_data_scores['mse']))
        print('test set mse (best_val):      ', lim(test_data_scores['mse']))
        print('test set MAE (best_val):      ', lim(test_data_scores['mae']))
        print('test set R2  (best_val):      ', lim(test_data_scores['r2']))
        print('test set RMSE(best_val):      ', lim(test_data_scores['rmse']))

    else:
        print('train/valid/test set accuracy (best_val): {} / {} / {}'.format(lim(training_data_scores['accuracy']), lim(validation_data_scores['accuracy']), lim(test_data_scores['accuracy'])))
        if 'auc' in test_data_scores:
            print('train/valid/test set AUC      (best_val): {} / {} / {}'.format(lim(training_data_scores['auc']), lim(validation_data_scores['auc']), lim(test_data_scores['auc'])))

    return model, (training_data_scores, validation_data_scores, test_data_scores), (log_train_loss, log_validation_loss), Targets_UnNormalization_fn(test_predictions)
















def _main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set', type=str, default=None,
                        help="data set selection (implemented by default in this example: 'delaney', 'huuskonsen', 'karthikeyan', 'bloodbrainbarrier'")

    parser.add_argument('--data_csv_file', type=str, default=None,
                        help="Not required if data_set is set. Specifies the path to the csv file holding the training data (SMILES + targets)")

    parser.add_argument('--csv_data_field', type=str, default='smiles',
                        help="Not required if data_set is set. Specifies the name of the field corresponding to the SMILES")

    parser.add_argument('--csv_targets_field', type=str, default=None,
                        help="Not required if data_set is set. Specifies the name of the field corresponding to the targets (e.g. 'solubility')")




    args = parser.parse_args()

    '''
        Two implementations are available (they are equivalent): index_based and matrix_based.
        The index_based one is usually slightly faster.
    '''



    my_data, my_labels, is_regression, _num_classes = train_helper.load_and_cache_csv(args.data_set)

    model, train_scores, val_scores, test_scores = perform_cross_validation(my_data, my_labels,
                                                                            CNN_model_hyperparamters = Default_CNN_model_hyperparamters,
                                                                            regression=is_regression,
                                                                            num_classes=_num_classes,
                                                                            use_matrix_based_implementation=False,
                                                                            plot_training_mse=False)


    # to save the model weights use:
    train_helper.save_model_weights(model, 'trained_fingerprint_model.npz')

    # to load the saved model weights use:
    train_helper.load_model_weights(model, 'trained_fingerprint_model.npz')


    #this saves an image of the network's computational graph (an abstract form of it)
    # beware that this requires the 'graphviz' software!
    train_helper.save_model_visualization(model, filename = 'fingerprintmodel.png')

    train_helper.pyplot.show()



    return


if __name__=='__main__':
    _main()


