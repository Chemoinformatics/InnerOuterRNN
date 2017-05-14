from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import matplotlib.pyplot as plt
import tensorflow as tf

import numpy as np

import config
from molecule import Molecule
import nn_utils
from utils import get_metric, save_results

logger = logging.getLogger(__name__)


class UGRNN(object):
    def __init__(self, model_name, encoding_nn_hidden_size, encoding_nn_output_size,
                 output_nn_hidden_size, batch_size=1, learning_rate=0.001,  add_logp=False, 
                 clip_gradients = False):
        """Build the ugrnn model up to where it may be used for inference."""

        # logger.info("Creating the UGRNN")
        # logger.info('Initial learning rate: {:}'.format(learning_rate))

        self.model_name = model_name
        self.batch_size = batch_size
        self._Unique_ID = str(self).split()[-1][:-1]
        self._clip_gradients = clip_gradients
        
        """Create placeholders"""
        self.local_input_pls = [tf.placeholder(tf.float32, shape=[None, None, Molecule.num_of_features()]) for i in
                                range(self.batch_size)]
        self.path_pls = [tf.placeholder(tf.int32, shape=[None, None, 3]) for i in range(self.batch_size)]
        self.target_pls = [tf.placeholder(tf.float32) for i in range(self.batch_size)]
        self.logp_pls = [tf.placeholder(tf.float32) for i in range(self.batch_size)]
        self.sequence_len_pls = [tf.placeholder(tf.int32) for i in range(self.batch_size)]
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.global_step_update_op = tf.assign(self.global_step, tf.add(self.global_step, tf.constant(1)))

        """Set the hyperparameters for the model"""
        self.learning_rate = learning_rate * tf.pow(config.learning_rate_decay_factor, tf.to_float(self.global_step),
                                                    name=None)
        self.encoding_nn_output_size = encoding_nn_output_size
        self.encoding_nn_hidden_size = encoding_nn_hidden_size
        self.encoding_nn_input_size = 4 * encoding_nn_output_size + Molecule.num_of_features()
        self.add_logp = add_logp

        if self.add_logp:
            self.output_nn_input_size = self.encoding_nn_output_size + 1
        else:
            self.output_nn_input_size = self.encoding_nn_output_size
        self.output_nn_hidden_size = output_nn_hidden_size

        self.initializer_fun = nn_utils.get_initializer(config.initializer)

        self.flattened_idx_offsets = [(tf.range(0, self.sequence_len_pls[i]) * config.max_seq_len * 4) for i in
                                      range(0, self.batch_size)]

        self.trainable_variables = []
        self.create_UGRNN_variable()

        prediction_op, g_structue = self.add_prediction_op(self.local_input_pls[0],
                                               self.path_pls[0],
                                               self.logp_pls[0],
                                               self.sequence_len_pls[0],
                                               self.flattened_idx_offsets[0])
        self.prediction_ops = [prediction_op]
        self.g_structure_ops = [g_structue]
        for i in range(1, self.batch_size):
            with tf.control_dependencies([self.prediction_ops[i - 1]]):
                prediction_op, g_structue = self.add_prediction_op(self.local_input_pls[i],
                                                       self.path_pls[i],
                                                       self.logp_pls[i],
                                                       self.sequence_len_pls[i],
                                                       self.flattened_idx_offsets[i])
                self.prediction_ops.append(prediction_op)
                self.g_structure_ops.append(g_structue)

        self.loss_op = self.add_loss_op()
        self.train_op = self.add_training_ops()




    def create_UGRNN_variable(self):
        with tf.variable_scope(self._Unique_ID+"EncodingNN") as scope:
            contextual_features = tf.get_variable("contextual_features",
                                                  [config.max_seq_len * config.max_seq_len * 4,
                                                   self.encoding_nn_output_size],
                                                  dtype=tf.float32,
                                                  initializer=tf.constant_initializer(0),
                                                  trainable=False)

            with tf.variable_scope('hidden1') as scope:
                weights = nn_utils.weight_variable([self.encoding_nn_input_size,
                                                 self.encoding_nn_hidden_size],
                                                initializer=self.initializer_fun)
                biases = nn_utils.bias_variable([self.encoding_nn_hidden_size])
                self.trainable_variables.append(weights)
                self.trainable_variables.append(biases)

            with tf.variable_scope('output') as scope:
                weights = nn_utils.weight_variable(
                    [self.encoding_nn_hidden_size,
                     self.encoding_nn_output_size],
                    initializer=self.initializer_fun)

                biases = nn_utils.bias_variable([self.encoding_nn_output_size])
                self.trainable_variables.append(weights)
                self.trainable_variables.append(biases)

        with tf.variable_scope(self._Unique_ID+"OutputNN") as scope:
            with tf.variable_scope('hidden1') as scope:
                weights = nn_utils.weight_variable(
                    [self.output_nn_input_size, self.output_nn_hidden_size],
                    self.initializer_fun, 'weights_decay')

                biases = nn_utils.bias_variable([self.output_nn_hidden_size])
                self.trainable_variables.append(weights)
                self.trainable_variables.append(biases)

            with tf.variable_scope('output') as scope:
                weights = nn_utils.weight_variable(
                    [self.output_nn_hidden_size, 1],
                    self.initializer_fun, 'weights_decay')
                self.trainable_variables.append(weights)

    def add_prediction_op(self, feature_pl, path_pl, logp_pl, sequence_len, flattened_idx_offset):
        with tf.variable_scope(self._Unique_ID+"EncodingNN", reuse=True) as scope:
            step = tf.constant(0)
            contextual_features = tf.get_variable("contextual_features")
            contextual_features = contextual_features.assign(
                tf.zeros([config.max_seq_len * config.max_seq_len * 4,
                          self.encoding_nn_output_size],
                         dtype=tf.float32))

            _, step, _, _, _, contextual_features, _ = tf.while_loop(
                UGRNN.cond, self.body,
                [sequence_len, step, feature_pl,
                 path_pl,
                 flattened_idx_offset, contextual_features,
                 self.encoding_nn_output_size],
                back_prop=True,
                swap_memory=False, name=None)

            # use flattened indices1
            step_contextual_features = UGRNN.get_contextual_feature(
                contextual_features=contextual_features,
                index=0,
                flattened_idx_offset=flattened_idx_offset,
                encoding_nn_output_size=self.encoding_nn_output_size)

            indices = tf.pack([tf.range(0, sequence_len), tf.range(0, sequence_len)], axis=1)
            step_feature = tf.gather_nd(feature_pl, indices)

            inputs = tf.concat(1, [step_contextual_features, step_feature])
            encodings = UGRNN.apply_EncodingNN(inputs, config.activation_function)

            molecule_encoding = tf.expand_dims(tf.reduce_sum(encodings, 0), 0)
            x = tf.expand_dims(logp_pl, 0)
            x = tf.expand_dims(x, 1)
            if self.add_logp:
                outputNN_input = tf.concat(1, [x, molecule_encoding])
            else:
                outputNN_input = molecule_encoding

        with tf.variable_scope(self._Unique_ID+"OutputNN", reuse=True) as scope:
            prediction_op = UGRNN.apply_OutputNN(outputNN_input,
                                                 config.activation_function)

        return prediction_op, molecule_encoding

    @staticmethod
    def cond(sequence_len, step, feature_pl, path_pl, flattened_idx_offset,
             contextual_features, encoding_nn_output_size):
        return tf.less(step, sequence_len - 1)

#    @staticmethod
    def body(self,sequence_len, step, feature_pl, path_pl, flattened_idx_offset,
             contextual_features, encoding_nn_output_size):
        zero = tf.constant(0)
        one = tf.constant(1)
        input_begin = tf.pack([zero, step, zero])

        input_idx = tf.slice(path_pl, input_begin, [-1, 1, 1])
        input_idx = tf.reshape(input_idx, [-1])

        indices = tf.pack([tf.range(0, sequence_len), input_idx], axis=1)
        step_feature = tf.gather_nd(feature_pl, indices)

        output_begin = tf.pack([zero, step, one])
        tf.get_variable_scope().reuse_variables()

        contextual_features = tf.get_variable("contextual_features")

        step_contextual_features = UGRNN.get_contextual_feature(
            contextual_features=contextual_features,
            index=input_idx,
            flattened_idx_offset=flattened_idx_offset,
            encoding_nn_output_size=encoding_nn_output_size)

        nn_inputs = tf.concat(1, [step_contextual_features, step_feature])
        updated_contextual_vectors = UGRNN.apply_EncodingNN(nn_inputs,
                                                            config.activation_function)
        output_idx = tf.squeeze(tf.slice(path_pl, output_begin, [-1, 1, 2]))

        contextual_features = UGRNN.update_contextual_features(
            contextual_features=contextual_features,
            indices=output_idx,
            updates=updated_contextual_vectors,
            flattened_idx_offset=flattened_idx_offset)

        with tf.control_dependencies([contextual_features]):
            return (sequence_len,
                    step + 1,
                    feature_pl,
                    path_pl,
                    flattened_idx_offset,
                    contextual_features,
                    encoding_nn_output_size)

    def add_training_ops(self):
        def apply_gradient_clipping(gradient):
            if gradient is not None:
                return tf.mul(tf.clip_by_value(tf.abs(grad), 0.1, 1.),
                              tf.sign(grad))
            else:
                return None

        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                           beta1=0.9, beta2=0.999,
                                           epsilon=1e-08,
                                           use_locking=False, name='Adam')

        loss_op = self.loss_op + config.weight_decay_factor * tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.get_collection('weights_decay')])

        gvs = optimizer.compute_gradients(loss_op)

        if self._clip_gradients:
            gvs = [(apply_gradient_clipping(grad), var) for grad, var in gvs]

        train_op = optimizer.apply_gradients(gvs)

        return train_op

    def add_loss_op(self):
        loss_op = [
            tf.square(tf.sub(self.prediction_ops[i], self.target_pls[i])) for i
            in range(0, self.batch_size)]
        loss_op = tf.add_n(loss_op, name=None) / 2

        return loss_op

    def train(self, sess, epochs, train_dataset, validation_dataset, output_dir, enable_plotting = 0):

        merged_summaries = tf.summary.merge_all()
        
        
        
        train_writer = tf.summary.FileWriter(output_dir + '/train', sess.graph)
        #train_writer = tf.train.SummaryWriter(output_dir + '/train', sess.graph)
#        print('train_dataset',train_dataset)

        train_metric = self.evaluate(sess, train_dataset)
        validation_metric = self.evaluate(sess, validation_dataset)
#        train_results_file_path = os.path.join(output_dir, "train_result.csv")
        
        if enable_plotting:
            plt.subplot(2, 1, 1)
            plt.title('Training data set')
            plt.axis([0, epochs, 0, train_metric[0]])
            plt.subplot(2, 1, 2)
            plt.title('Vaidation data set')
            plt.axis([0, epochs, 0, validation_metric[0]])
            plt.ion()
        logger.info('Start Training')

        steps_in_epoch = train_dataset.num_examples // self.batch_size
        # self.get_g_structure(sess,train_dataset)

        for epoch in range(0, epochs):
            for i in range(0, steps_in_epoch):
                feed_dict = self.fill_feed_dict(train_dataset, self.batch_size)
                _= sess.run([self.train_op], feed_dict=feed_dict)

            summ = sess.run(merged_summaries)
            train_writer.add_summary(summ, epoch)

            train_dataset.reset_epoch(permute=True)

            sess.run([self.global_step_update_op])

            if epoch % 10 == 0:
                train_metric = self.evaluate(sess, train_dataset)
                validation_metric = self.evaluate(sess, validation_dataset)
                if enable_plotting:
                    plt.subplot(2, 1, 1)
                    plt.scatter(epoch, train_metric[0], color='red', marker=".")
                    plt.scatter(epoch, train_metric[1], color='blue', marker=".")
                    plt.subplot(2, 1, 2)
                    plt.scatter(epoch, validation_metric[0], color='red', marker=".")
                    plt.scatter(epoch, validation_metric[1], color='blue', marker=".")
                    plt.pause(0.05)
                learning_rate = self.get_learning_rate(sess)
                
                logger.info(
                    "Epoch: {:}, Learning rate {:.8f}  Train RMSE: {:.4f}, Train AAE: {:.4f} Validation RMSE {:.4f}, Validation AAE {:.4f}".
                        format(epoch, learning_rate[0], train_metric[0],
                               train_metric[1], validation_metric[0],
                               validation_metric[1],
                               precision=8))

        save_results(output_dir, train_dataset.labels, self.predict(sess, train_dataset), additional_str='_train')
        save_results(output_dir, validation_dataset.labels, self.predict(sess, validation_dataset), additional_str='_valid')
        logger.info('Training Finished')

    def evaluate(self, sess,  dataset):
        predictions = self.predict(sess, dataset)
        targets = dataset.labels
        return get_metric(predictions, targets)

    def predict(self, sess, dataset):
        dataset.reset_epoch()
        predictions = np.empty(dataset.num_examples)
        for i in range(dataset.num_examples):
#            try:
            feed_dict = self.fill_feed_dict(dataset, 1)
#            except:
#                print('predict:: skipped @',i,'of',dataset.num_examples)
#                continue
#                return predictions[:i]
            prediction_value = sess.run([self.prediction_ops[0]], feed_dict=feed_dict)
            predictions[i] = np.mean(prediction_value)
        return predictions

    def get_g_structure(self,sess,dataset):
        dataset.reset_epoch()
#        g_structures = np.empty((dataset.num_examples, self.encoding_nn_output_size))
        for i in range(dataset.num_examples):
            feed_dict = self.fill_feed_dict(dataset, 1)
            g_structure = sess.run([self.g_structure_ops[0]], feed_dict=feed_dict)
            print(g_structure)
            # g_structures[i,:] = g_structure
        return g_structure

    def fill_feed_dict(self, dataset, batch_size):
        assert batch_size <= self.batch_size
        molecules_feeds, targets_feeds = dataset.next_batch(batch_size)
        
        feed_dict = {}
        for i in range(batch_size):
            
            feed_dict[self.local_input_pls[i]] = molecules_feeds[i].local_input_vector
            feed_dict[self.path_pls[i]] = molecules_feeds[i].directed_graphs
            feed_dict[self.target_pls[i]] = targets_feeds[i]
            feed_dict[self.sequence_len_pls[i]] = molecules_feeds[i].local_input_vector.shape[1]

            if self.add_logp:
                feed_dict[self.logp_pls[i]] = molecules_feeds[i].logp

        return feed_dict

    def get_learning_rate(self, sess):
        return sess.run([self.learning_rate])

    def save_model(self, sess, checkpoint_dir, step):
        logging.info("Saving model {:}".format(self.model_name))
        saver = tf.train.Saver(self.trainable_variables, max_to_keep=1)
        checkpoint_file = os.path.join(checkpoint_dir, 'model.ckpt')
        saver.save(sess, save_path=checkpoint_file)

    def restore_model(self, sess, checkpoint_dir):
        # logging.info("Restoring model {:}".format(self.model_name))
        saver = tf.train.Saver(self.trainable_variables)
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

    @staticmethod
    def get_contextual_feature(contextual_features, index,
                               flattened_idx_offset, encoding_nn_output_size):
        """
        Contextual vector is flatted array
            index is 1D index with
        """
        indices = index + flattened_idx_offset
        values = [indices, indices, indices, indices]
        indices = tf.pack(values, axis=1, name='pack')
        indices = indices + tf.constant([0, 1, 2, 3])
        indices = tf.reshape(indices, [-1])
        contextual_vector = tf.gather(contextual_features, indices)
        contextual_vector = tf.reshape(contextual_vector,
                                       [-1, 4 * encoding_nn_output_size])
        return contextual_vector

    @staticmethod
    def update_contextual_features(contextual_features, indices, updates,
                                   flattened_idx_offset):
        first_indices, second_indices = tf.split(1, 2, indices)
        indices = tf.squeeze(first_indices + second_indices)
        indices = indices + flattened_idx_offset
        contextual_features = tf.scatter_add(contextual_features, indices,
                                             updates, use_locking=None)
        return contextual_features

    @staticmethod
    def apply_EncodingNN(inputs, activation_type):
        activation_fun = nn_utils.get_activation_fun(activation_type)
        with tf.variable_scope('hidden1') as scope:
            weights = tf.get_variable("weights")
            biases = tf.get_variable("biases")

            hidden1 = activation_fun(tf.matmul(inputs, weights) + biases)

        with tf.variable_scope('output') as scope:
            weights = tf.get_variable("weights")
            biases = tf.get_variable("biases")
            return activation_fun(tf.matmul(hidden1, weights) + biases)

    @staticmethod
    def apply_OutputNN(inputs, activation_type):
        activation_fun = nn_utils.get_activation_fun(activation_type)
        with tf.variable_scope('hidden1') as scope:
            weights = tf.get_variable("weights")
            biases = tf.get_variable("biases")
            hidden1 = activation_fun(tf.matmul(inputs, weights) + biases)

        with tf.variable_scope('output') as scope:
            weights = tf.get_variable("weights")
            return tf.matmul(hidden1, weights)
