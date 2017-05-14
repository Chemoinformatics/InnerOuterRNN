from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

import argparse
import logging
import os

import numpy as np

from input_data import DataSet
from ugrnn import UGRNN
from utils import model_params, get_metric, save_results

np.set_printoptions(threshold=np.inf)

import tensorflow as tf

FLAGS = None

def get_prediction_from_model(model_name, encoding_nn_hidden_size, encoding_nn_output_size,
                              output_nn_hidden_size, test_dataset, validation_dataset):
    model_dir = os.path.join(FLAGS.output_dir, model_name)

    if not tf.gfile.Exists(model_dir):
        raise Exception("Invalid path or the model paramter doesnot exist")

    with tf.Graph().as_default():
        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # logger.info("Creating Graph.")

        ugrnn_model = UGRNN(model_name, encoding_nn_hidden_size=encoding_nn_hidden_size,
                            encoding_nn_output_size=encoding_nn_output_size,
                            output_nn_hidden_size=output_nn_hidden_size,
                            add_logp=FLAGS.add_logp)

        # logger.info("Succesfully created graph.")

        init = tf.global_variables_initializer()
        sess.run(init)
        # logger.info('Run the Op to initialize the variables')

        # logger.info('Restoring model parameters')
        ugrnn_model.restore_model(sess, model_dir)

        prediction_validate = ugrnn_model.predict(sess, validation_dataset)
        prediction_test = ugrnn_model.predict(sess, test_dataset)

    test_results_file_path = os.path.join(model_dir, "test_result.csv")
    validation_results_file_path = os.path.join(model_dir, "validation_result.csv")

    save_results(test_results_file_path, test_dataset.labels, prediction_test)
    save_results(validation_results_file_path, validation_dataset.labels, prediction_validate)

    return prediction_validate, prediction_test


def ensemble_prediction_linear_regression(validation_dataset, all_validation_predictions, all_test_predictions):
    lr = linear_model.LinearRegression(fit_intercept=False)
    lr.fit(all_validation_predictions.T, validation_dataset.labels)
    emsemble_preditions = lr.predict(all_test_predictions.T)
    # print("Liner Regression Weights: ", lr.coef_)
    return emsemble_preditions


def ensemble_prediction_rf_regression(validation_dataset, all_validation_predictions, all_test_predictions):
    rfr = RandomForestRegressor(n_estimators=1000)
    rfr.fit(all_validation_predictions.T, validation_dataset.labels)
    emsemble_preditions = rfr.predict(all_test_predictions.T)
    return emsemble_preditions


def ensemble_prediction_average(validation_dataset, all_validation_predictions, all_test_predictions):
    emsemble_preditions = np.mean(all_test_predictions, axis=0)
    return emsemble_preditions


def ensemble_prediction_top_k(validation_dataset, all_validation_predictions, all_test_predictions, k=10):
    no_of_models = len(all_validation_predictions)
    errors = []
    for i in xrange(0, no_of_models):
        metric = get_metric(all_validation_predictions[i], validation_dataset.labels)
        errors.append(metric[0])

    errors = np.array(errors)
    index_of_best_networks = errors.argsort()[:k]
    # logging.info("Top {:} models: {:}".format(k, index_of_best_networks))
    emsemble_preditions = np.mean(all_test_predictions[index_of_best_networks], axis=0)
    return emsemble_preditions

def ensemble_prediction_greedy(validation_dataset, all_validation_predictions, all_test_predictions):
    current_prediction = np.zeros(len(all_validation_predictions[0]))
    index = 0
    index_of_best_networks = []

    index_of_next_best = get_next_best_model(index, current_prediction,all_validation_predictions,validation_dataset.labels)

    while index_of_next_best != -1:
        index_of_best_networks.append(index_of_next_best)
        current_prediction = (index * current_prediction + all_validation_predictions[index_of_next_best]) / (index + 1)
        index+=1
        index_of_next_best = get_next_best_model(index, current_prediction, all_validation_predictions,
                                                 validation_dataset.labels)

    logging.info("Best models: {:}".format(index_of_best_networks))
    emsemble_preditions = np.mean(all_test_predictions[index_of_best_networks], axis=0)
    return emsemble_preditions


def get_next_best_model(index, current_prediction, all_predictions, targets):
    no_of_models = len(all_predictions)

    current_error = (get_metric(current_prediction, targets))[0]
    next_best_model_index = -1

    for i in xrange(0, no_of_models):
        temp_prediction = (index * current_prediction + all_predictions[i]) / (index + 1)
        metric = get_metric(temp_prediction, targets)
        if metric[0] < current_error:
            next_best_model_index = i
            current_error = metric[0]

    return next_best_model_index


def main(_):
    logger.info('Loading Models From {:}'.format(FLAGS.output_dir))

    logp_col_name = FLAGS.logp_col if FLAGS.add_logp else None
    test_dataset = DataSet(csv_file_path=FLAGS.test_file, smile_col_name=FLAGS.smile_col,
                           target_col_name=FLAGS.target_col, logp_col_name=logp_col_name,
                           contract_rings=FLAGS.contract_rings)


    validation_dataset = DataSet(csv_file_path=FLAGS.validation_file, smile_col_name=FLAGS.smile_col,
                                 target_col_name=FLAGS.target_col, logp_col_name=logp_col_name,
                                 contract_rings=FLAGS.contract_rings)

    validation_predictions = np.empty((len(FLAGS.model_names), validation_dataset.num_examples))
    test_predictions_ = np.empty((len(FLAGS.model_names), test_dataset.num_examples))

    for i in xrange(0, len(FLAGS.model_names)):
        predictions = get_prediction_from_model(FLAGS.model_names[i], FLAGS.model_params[i][0],
                                                FLAGS.model_params[i][1], FLAGS.model_params[i][2],
                                                test_dataset, validation_dataset)

        validation_predictions[i, :] = predictions[0]
        test_predictions_[i, :] = predictions[1]
    ensemble_predictor = [ensemble_prediction_rf_regression, ensemble_prediction_top_k, ensemble_prediction_greedy]
    predictor_names = [ "Random forest regression", "Top 10", "Greedy"]

    for fun, name in zip(ensemble_predictor, predictor_names):
        emsemble_preditions = fun(validation_dataset, validation_predictions,  test_predictions_)
        prediction_metric = get_metric(emsemble_preditions, test_dataset.labels)
        logger.info("Method {:} RMSE: {:}, AAE: {:}, R: {:}".format(name, prediction_metric[0], prediction_metric[1],
                                                                   prediction_metric[2]))

    final_prediction_path = os.path.join(FLAGS.output_dir, "ensemble_test_prediction.csv")
    save_results(final_prediction_path, test_dataset.labels, emsemble_preditions)
    logging.info("------------------------------DONE------------------------------")
    logging.info("")
    logging.info("")

if __name__ == '__main__':
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_names', nargs='+', type=str,
                        help='Name of the models used for prediction')

    parser.add_argument('--model_params', help="Model Parameters", dest="model_params", type=model_params, nargs='+')

    parser.add_argument('--output_dir', type=str, default='train',
                        help='Root Directory where the  model parameters are stored')

    parser.add_argument('--test_file', type=str, default='ugrnn/data/delaney/validate_delaney.csv',
                        help='Path to the csv file containing test data set')

    parser.add_argument('--validation_file', type=str, default='ugrnn/data/delaney/test_delaney.csv',
                        help='Path to the csv file containing validation data set')

    parser.add_argument('--smile_col', type=str, default='smiles')

    parser.add_argument('--logp_col', type=str, default='logp')

    parser.add_argument('--target_col', type=str, default='solubility')

    parser.add_argument('--contract_rings', dest='contract_rings',
                        action='store_true')
    parser.set_defaults(contract_rings=False)

    parser.add_argument('--add_logp', dest='add_logp',
                        action='store_true')
    parser.set_defaults(add_logp=False)

    parser.add_argument('--optimize_ensemble', dest='optimize_ensemble',
                        action='store_true')
    parser.set_defaults(optimize_ensemble=False)

    FLAGS = parser.parse_args()
    assert len(FLAGS.model_params) == len(FLAGS.model_names)

    tf.app.run(main=main)


