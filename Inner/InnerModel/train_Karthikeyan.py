from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os

import numpy as np

from input_data import DataSet
from ugrnn import UGRNN
from utils import model_params
np.set_printoptions(threshold=np.inf, precision=4)

import tensorflow as tf
import utils 

FLAGS = None




def run_once(session, output_dir, train_data, valid_data, logp_col_name, experiment_name = ''):
    
#    logp_col_name=logp_col_name,
    train_dataset = DataSet(smiles=train_data[0], labels=train_data[1], contract_rings=FLAGS.contract_rings)
    validation_dataset = DataSet(smiles=valid_data[0], labels=valid_data[1],  contract_rings=FLAGS.contract_rings)



    logger.info("Creating Graph.")
    ugrnn_model = UGRNN(FLAGS.model_name, encoding_nn_hidden_size=FLAGS.model_params[0],
                        encoding_nn_output_size=FLAGS.model_params[1], output_nn_hidden_size=FLAGS.model_params[2],
                        batch_size=FLAGS.batch_size, learning_rate=0.001, add_logp=FLAGS.add_logp, 
                        clip_gradients=FLAGS.clip_gradient)
    logger.info("Succesfully created graph.")
    
    init = tf.global_variables_initializer()
    session.run(init)
    logger.info('Run the Op to initialize the variables')
    print('FLAGS.enable_plotting',FLAGS.enable_plotting)
    ugrnn_model.train(session, FLAGS.max_epochs, train_dataset, validation_dataset, output_dir, enable_plotting = int(FLAGS.enable_plotting))
    ugrnn_model.save_model(session, output_dir, FLAGS.max_epochs)








def main(*args):
    output_dir = os.path.join(FLAGS.output_dir, FLAGS.model_name)

#    if tf.gfile.Exists(output_dir):
#        tf.gfile.DeleteRecursively(output_dir)
    
    tf.gfile.MakeDirs(output_dir)
    

    with tf.Graph().as_default():
        # Create a session for running Ops on the Graph.
        session = tf.Session()

        logp_col_name = FLAGS.logp_col if FLAGS.add_logp else None
        
        
        
        logger.info('Loading data set from {:}'.format(FLAGS.training_file))
        csv_file_path=FLAGS.training_file
        smile_col_name=FLAGS.smile_col
        target_col_name=FLAGS.target_col
        data = utils.read_csv(csv_file_path, smile_col_name, target_col_name, logp_col_name)
        data = list(zip(*data))
        
        if FLAGS.validation_file!='':
            logger.info('Loading validation dataset from {:}'.format(FLAGS.validation_file))
            valid_data = utils.read_csv(FLAGS.validation_file, smile_col_name, target_col_name, logp_col_name)
            train_data = data
            
            run_once(session, output_dir, list(zip(*train_data)), list(zip(*valid_data)), logp_col_name)
            
        else:
            assert FLAGS.initial_crossvalidation_index <FLAGS.crossval_total_num_splits, 'INVALID VALUE GIVEN!'
            for crossval_split_index in range(FLAGS.initial_crossvalidation_index, FLAGS.crossval_total_num_splits):
                print('crossval_split: {} of {}'.format(crossval_split_index+1, FLAGS.crossval_total_num_splits))
                
                assert len(data[0])==len(data[1])
                train_data, valid_data, testdata = utils.cross_validation_split(data[0], data[1], crossval_split_index, crossval_total_num_splits=FLAGS.crossval_total_num_splits, validation_data_ratio=1./FLAGS.crossval_total_num_splits)
                #merge "test" and train -- validation part used for testing
                train_data = (np.concatenate((train_data[0], testdata[0])), np.concatenate((train_data[1], testdata[1])))
                print('CV: # train samples:',len(train_data[0]),'# validation samples:', len(valid_data[0]))
                #to list of tuples
#                train_data = list(zip(*train_data))
#                valid_data = list(zip(*valid_data))
#                print(train_data)
                run_once(session, output_dir+'_CV_{}'.format(crossval_split_index), train_data, valid_data, logp_col_name)
            
            
        


if __name__ == '__main__':
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='default_karthikeyan_model',
                        help='Name of the model')

    parser.add_argument('--max_epochs', type=int, default=120,
                        help='Number of epochs to run trainer.')

    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size.')

    parser.add_argument('--model_params', help="Model Parameters", dest="model_params", type=model_params, default = '7,7,5')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')

    parser.add_argument('--output_dir', type=str, default='train',
                        help='Directory for storing the trained models')

    parser.add_argument('--training_file', type=str, default='../data/karthikeyan/melting_points.csv',
                        help='Path to the csv file containing training data set')

    parser.add_argument('--validation_file', type=str, default='', #'data/delaney/validate_delaney.csv',
                        help='Path to the csv file containing validation data set (if not provided, then a cross-validation is performed)')

    parser.add_argument('--crossval_total_num_splits', type=int, default=10)

    parser.add_argument('--smile_col', type=str, default='SMILES')

    parser.add_argument('--logp_col', type=str, default='logp')

    parser.add_argument('--target_col', type=str, default='MTP')

    parser.add_argument('--contract_rings', dest='contract_rings',default = False)

    parser.add_argument('--add_logp', dest='add_logp', default = False)
    
    parser.add_argument('--clip_gradient', dest='clip_gradient', default=False)
    
    parser.add_argument('--enable_plotting', dest='enable_plotting', default=False)
    
    parser.add_argument('--initial_crossvalidation_index', type=int, default=9)
    
    

    FLAGS = parser.parse_args()
    
    main()
    #tf.app.run(main=main)
