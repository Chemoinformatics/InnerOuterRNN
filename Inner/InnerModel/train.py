from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import numpy as np


try:
    from . import config
    from . import utils
    from .input_data import DataSet
    from .ugrnn import UGRNN
    from .utils import model_params_formatting
except:
    import config
    import utils
    from input_data import DataSet
    from ugrnn import UGRNN
    from utils import model_params_formatting


np.set_printoptions(threshold=np.inf, precision=4)
import tensorflow as tf









def build_and_train(logger, session, output_dir, train_data, valid_data, experiment_name = '',
                    regression = True, binary_classification = False, model_name = 'ugrnn_1',
                    batch_size = 10, clip_gradient = False, model_params = None,
                    contract_rings = False, learning_rate = 1e-3, max_epochs = 150, enable_plotting = False,
                    Targets_UnNormalization_fn = lambda x:x, weight_decay_factor = 0, *args, **kwargs):

    # TODO: figure out what causes the internal Tensorflow bug that requires this hack ('remove_SMILES_longer_than').
    # is it due to a new ("improved") tensorflow version?
    train_data = utils.remove_SMILES_longer_than(train_data, config.max_seq_len)
    valid_data = utils.remove_SMILES_longer_than(valid_data, config.max_seq_len)

    train_labels, is_masked_t = utils.create_labels_NaN_mask(train_data[1])
    valid_labels, is_masked_v = utils.create_labels_NaN_mask(valid_data[1])

    # inferring stuff based on the data

    is_masked = is_masked_t or is_masked_v
    multitask = (not regression) and binary_classification
    num_tasks = train_labels.shape[-1] if train_labels.ndim>1 else 1


    assert not (regression and binary_classification), 'ERROR: arguments <regression>==True and <binary_classification>==True are mutually exclusive.'

    if is_masked:
        if not is_masked_t:
            train_labels, is_masked_t = utils.create_labels_NaN_mask(train_data[1], force_masked=1)
        if not is_masked_v:
            valid_labels, is_masked_v = utils.create_labels_NaN_mask(valid_data[1], force_masked=1)


    train_dataset      = DataSet(smiles=train_data[0], labels=train_labels, contract_rings=contract_rings)
    validation_dataset = DataSet(smiles=valid_data[0], labels=valid_labels, contract_rings=contract_rings)


    logger.info("Creating Graph.")
    ugrnn_model = UGRNN(model_name, encoding_nn_hidden_size=model_params[0],
                        encoding_nn_output_size=model_params[1], output_nn_hidden_size=model_params[2],
                        batch_size=batch_size, learning_rate=learning_rate, add_logp=False,
                        clip_gradients=clip_gradient, regression = regression, weight_decay_factor = weight_decay_factor,
                        num_tasks = num_tasks, multitask = multitask, weighted_loss = is_masked)
    logger.info("Succesfully created graph.")

    init = tf.global_variables_initializer()
    session.run(init)

    training_scores_dict, validation_scores_dict = ugrnn_model.train(session, max_epochs, train_dataset, validation_dataset,
                                                                     output_dir, enable_plotting = bool(enable_plotting),
                                                                     Targets_UnNormalization_fn = Targets_UnNormalization_fn)
    ugrnn_model.save_model(session, output_dir, max_epochs)
    return training_scores_dict, validation_scores_dict








def main(output_dir = 'output/', model_name = 'my_model',
         training_file = 'delaney_train.csv', validation_file = 'delaney_validate.csv', smile_col = 'smiles',
         target_col = 'solubility', crossval_total_num_splits = 10, initial_crossvalidation_index = 0,
         weight_decay_factor = 0, *args, **kwargs):
    '''
    valid kwargs:

        experiment_name, regression,
        binary_classification, batch_size,
        clip_gradient, model_params,
        contract_rings, learning_rate,
        max_epochs, enable_plotting

    '''
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger(__name__)
    print('output_dir',output_dir)
    output_dir = os.path.join(output_dir, model_name)

#    if tf.gfile.Exists(output_dir):
#        tf.gfile.DeleteRecursively(output_dir)

    tf.gfile.MakeDirs(output_dir)


    with tf.Graph().as_default():
        # Create a session for running Ops on the Graph.
        # select CPU (as it is faster than GPUs)
        config = tf.ConfigProto(device_count = {'GPU': 0})
        session = tf.Session(config=config)



        logger.info('Loading data set from {:}'.format(training_file))
        csv_file_path=training_file
        smile_col_name=smile_col
        target_col_name=target_col
        data = utils.read_csv(csv_file_path, None, smile_col_name, target_col_name)
        assert len(data[0])>0, 'no data loaded!'
        smiles, labels = utils.permute_data(data[0], data[1])

        if kwargs['regression']:
            # normalize regression targets to be in a reasonable value-range
            labels_mean  = labels.mean()
            labels_range = np.max(labels) - np.min(labels)
            labels = (labels - labels_mean)/labels_range
            #this function will be applied to predictions of the model and to targets when computing metrics
            def Targets_UnNormalization_fn(targets):
                return targets*labels_range + labels_mean
            def Targets_Normalization_fn(targets):
                return (targets - labels_mean)/labels_range
        else:
            if labels.ndim==1:
                labels = labels.reshape((len(labels),1))
            Targets_UnNormalization_fn = lambda x:x
            Targets_Normalization_fn   = lambda x:x



        if validation_file!='' and validation_file is not None:
            # train single model
            logger.info('Loading validation dataset from {:}'.format(validation_file))
            valid_data = utils.read_csv(validation_file, None, smile_col_name, target_col_name)
            if kwargs['regression']==0 and labels.ndim==1:
                labels = labels.reshape((len(labels),1)) #binary classification
            train_data = (smiles, labels)
            valid_data = (valid_data[0], Targets_Normalization_fn(valid_data[1]))

            training_scores_dict, validation_scores_dict = build_and_train(logger, session, output_dir, train_data, valid_data,
                                                     model_name = model_name,
                                                     Targets_UnNormalization_fn = Targets_UnNormalization_fn,
                                                     weight_decay_factor = weight_decay_factor, **kwargs)

        else:
            # cross validation
            assert initial_crossvalidation_index <crossval_total_num_splits, 'INVALID VALUE GIVEN for initial_crossvalidation_index or crossval_total_num_splits!'
            training_scores_dict, validation_scores_dict = [], []
            for crossval_split_index in range(initial_crossvalidation_index, crossval_total_num_splits):
                print('crossval_split: {} of {}'.format(crossval_split_index+1, crossval_total_num_splits))

                assert len(smiles)==len(labels)
                train_data, valid_data, testdata = utils.cross_validation_split(smiles, labels, crossval_split_index, crossval_total_num_splits=crossval_total_num_splits, validation_data_ratio=1./crossval_total_num_splits)
                #merge "test" and train -- validation part used for testing
                train_data = (np.concatenate((train_data[0], testdata[0])), np.concatenate((train_data[1], testdata[1])))
                print('CV: # train samples:',len(train_data[0]),'# validation samples:', len(valid_data[0]))

                td, vd = build_and_train(logger, session,
                                         output_dir+'_CV_{}'.format(crossval_split_index),
                                         train_data, valid_data, model_name = model_name,
                                         Targets_UnNormalization_fn = Targets_UnNormalization_fn,
                                         weight_decay_factor = weight_decay_factor, **kwargs)
                training_scores_dict.append(td)
                validation_scores_dict.append(vd)
        if isinstance(training_scores_dict,list) and len(training_scores_dict)==1 and len(validation_scores_dict)==1:
            return training_scores_dict[0], validation_scores_dict[0]
        return training_scores_dict, validation_scores_dict



if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='default_model',
                        help='Name of the model')

    parser.add_argument('--max_epochs', type=int, default=120,
                        help='Number of epochs to run trainer.')

    parser.add_argument('--batch_size', type=int, default=6,
                        help='Batch size.')

    parser.add_argument('--model_params', help="Model Parameters (Three values, corresponding to the size of the: 1) crawling network; 2) size of the crawling output layer; and 3) size of the prediction network layer)",
                        dest="model_params", type=model_params_formatting, default = '14,12,13')

    parser.add_argument('--learning_rate', type=float, default=0.00061,
                        help='Initial learning rate')

    parser.add_argument('--output_dir', type=str, default='model_training_output',
                        help='Directory for storing the trained models')

    parser.add_argument('--training_file', type=str, default='../data/delaney/train_delaney.csv',
                        help='Path to the csv file containing training data set')

    parser.add_argument('--validation_file', type=str, default='',
                        help='Path to the csv file containing validation data set (if not provided, then a cross-validation is performed)')

    parser.add_argument('--crossval_total_num_splits', help='number of cross validation splits; a higher value creates larger training sub-sets and thus usually increases overall model performance but the cross-validation process will take longer.',
                        type=int, default=10)

    parser.add_argument('--smile_col', type=str, default='smiles')

    #parser.add_argument('--logp_col', type=str, default='logp')

    parser.add_argument('--target_col', type=str, default='solubility', help='name of the column containing the target(s) for prediction. You can specify multiple targets separated by a comma ","')

    parser.add_argument('--contract_rings', dest='contract_rings',default = False)

    #parser.add_argument('--add_logp', dest='add_logp', default = False)

    parser.add_argument('--clip_gradient', dest='clip_gradient', default=False)

    parser.add_argument('--enable_plotting', dest='enable_plotting', default=False)

    parser.add_argument('--initial_crossvalidation_index', type=int, default=0)

    parser.add_argument('--regression', type=bool, default=True)

    flags_ = vars(parser.parse_args())
    print('\nInitiating training with arguments:')
    prnt = sorted(['{} = <{}>'.format(x, str(v)) for x,v in flags_.items()])
    print('\n'.join(prnt))
    print()
    main(**flags_)
    #tf.app.run(main=main)
