from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
#import time

import matplotlib.pyplot as plt
import tensorflow as tf

try: #compatibility with multiple TF versions
    tf.pack = tf.stack
    tf.unpack = tf.unstack
    tf.sub = tf.subtract
except:
    pass #
import numpy as np


try:
    from . import config
    from .molecule import Molecule
    from . import nn_utils
    from .utils import get_metric, save_results
except:
    import config
    from molecule import Molecule
    import nn_utils
    from utils import get_metric, save_results



logger = logging.getLogger(__name__)


class UGRNN(object):
    def __init__(self, model_name, encoding_nn_hidden_size, encoding_nn_output_size,
                 output_nn_hidden_size, batch_size=1, learning_rate=0.001,
                 regression = True, multitask = False, num_tasks = 1, weighted_loss = False,
                 add_logp=False, clip_gradients = False, weight_decay_factor = 0):
        """Build the ugrnn-/inner model.

        regression:

            if False then a classification model is built

        multitask:

            only effective for classification, will assume that each task is binary classification

        weighted_loss:

            only effective for multitask classification; labels are assumed to be a 2-tensor of shape (2, num_tasks), i.e. [labels, weights].

        """

        # logger.info("Creating the UGRNN")
        # logger.info('Initial learning rate: {:}'.format(learning_rate))

        self.model_name = model_name
        self.batch_size = batch_size
        self._Unique_ID = str(self).split()[-1][:-1]
        self._clip_gradients = clip_gradients
        self._multitask = multitask
        self._weight_decay_factor = weight_decay_factor

        """Create placeholders"""
        self.local_input_pls = [tf.placeholder(tf.float32, shape=[None, None, Molecule.num_of_features()]) for i in
                                range(self.batch_size)]
        self.path_pls = [tf.placeholder(tf.int32, shape=[None, None, 3]) for i in range(self.batch_size)]
        if num_tasks>1:
            if weighted_loss:
                self.target_pls = [tf.placeholder(tf.float32, shape=[2, num_tasks]) for i in range(self.batch_size)]
            else:
                self.target_pls = [tf.placeholder(tf.float32, shape=[num_tasks]) for i in range(self.batch_size)]
        else:
            self.target_pls = [tf.placeholder(tf.float32) for i in range(self.batch_size)]
        self.logp_pls = [tf.placeholder(tf.float32) for i in range(self.batch_size)]
        self.sequence_len_pls = [tf.placeholder(tf.int32) for i in range(self.batch_size)]
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.global_step_update_op = tf.assign(self.global_step, tf.add(self.global_step, tf.constant(1)))

        """Set the hyperparameters for the model"""
        self.learning_rate = learning_rate * tf.pow(config.learning_rate_decay_factor, tf.to_float(self.global_step), name=None)
        self.encoding_nn_output_size = encoding_nn_output_size
        self.encoding_nn_hidden_size = encoding_nn_hidden_size
        self.encoding_nn_input_size = 4 * encoding_nn_output_size + Molecule.num_of_features()
        self.add_logp = add_logp

        if self.add_logp:
            self.output_nn_input_size = self.encoding_nn_output_size + 1
        else:
            self.output_nn_input_size = self.encoding_nn_output_size
        self.output_nn_hidden_size = output_nn_hidden_size
        self.output_nn_num_outputs = num_tasks

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

        if regression or (not multitask):
            self.loss_op = self.add_loss_op__MSE()
            assert not weighted_loss, 'weighted loss not (currently) supported for regression'
            if not regression:
                print('Warning: will use MSE for (assumed) binary classification task')
        else:

            if multitask and weighted_loss:
                self.loss_op = self.add_loss_op__binary_crossentropy_weighted()
            elif multitask:
                self.loss_op = self.add_loss_op__binary_crossentropy()
            else:
                raise NotImplementedError()

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
                    [self.output_nn_hidden_size, self.output_nn_num_outputs],
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

            inputs = tf.concat(axis=1, values=[step_contextual_features, step_feature])
            encodings = UGRNN.apply_EncodingNN(inputs, config.activation_function)

            molecule_encoding = tf.expand_dims(tf.reduce_sum(encodings, 0), 0)
            x = tf.expand_dims(logp_pl, 0)
            x = tf.expand_dims(x, 1)
            if self.add_logp:
                outputNN_input = tf.concat(axis=1, values=[x, molecule_encoding])
            else:
                outputNN_input = molecule_encoding
        output_act_fn = 'linear'
        if self._multitask:
            output_act_fn = 'sigmoid'
        with tf.variable_scope(self._Unique_ID+"OutputNN", reuse=True) as scope:
            prediction_op = UGRNN.apply_OutputNN(outputNN_input, config.activation_function, output_act_fn)

        return prediction_op, molecule_encoding

    @staticmethod
    def cond(sequence_len, step, feature_pl, path_pl, flattened_idx_offset,
             contextual_features, encoding_nn_output_size):
        return tf.less(step, sequence_len - 1)

#    @staticmethod
    def body(self,sequence_len, step, feature_pl, path_pl, flattened_idx_offset, contextual_features, encoding_nn_output_size):
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

        nn_inputs = tf.concat(axis=1, values=[step_contextual_features, step_feature])
        updated_contextual_vectors = UGRNN.apply_EncodingNN(nn_inputs, config.activation_function)
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
                return tf.mul(tf.clip_by_value(tf.abs(grad), 0.1, 1.), tf.sign(grad))
            else:
                return None
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                           beta1=0.9, beta2=0.999,
                                           epsilon=1e-08,
                                           use_locking=False, name='Adam')
        loss_op = self.loss_op
        if self._weight_decay_factor:
            loss_op = loss_op + self._weight_decay_factor * tf.add_n([tf.nn.l2_loss(v) for v in tf.get_collection('weights_decay')])
        gvs = optimizer.compute_gradients(loss_op)
        if self._clip_gradients:
            gvs = [(apply_gradient_clipping(grad), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(gvs)
        return train_op


    #add_loss_op
    def add_loss_op__MSE(self):
        '''
        expecting labels to be a list of 0-tensors/scalars: [(1,), ...] (length == batch_size)

        '''
        loss_op = [tf.square(tf.sub(self.prediction_ops[i], self.target_pls[i])) for i in range(0, self.batch_size)]
        loss_op = tf.add_n(loss_op, name=None) / self.batch_size #2
        return loss_op


    def add_loss_op__binary_crossentropy(self, eps = 1e-6):
        '''
        expecting labels to be a list of 1-tensors: [(num_classes,), ...] (length == batch_size)

        '''
        def my_binary_crossentropy(y_true, y_pred, eps_ = eps):
            return -(y_true*tf.log(y_pred + eps_) + (1-y_true)*tf.log((1-y_pred) + eps_))
        loss_op = [tf.reduce_mean(my_binary_crossentropy(self.target_pls[i], self.prediction_ops[i]), axis=None, keep_dims=False) for i in range(0, self.batch_size)]
        loss_op = tf.add_n(loss_op, name=None) / self.batch_size #2
        return loss_op


    def add_loss_op__binary_crossentropy_weighted(self, eps = 1e-6):
        '''
        expecting labels to be a list of 2-tensors: [(2, num_classes), ...] (length == batch_size)

        where [:,0,:] selects the labels, and [:,1,:] selects the mask/weights.

        '''
        def my_binary_crossentropy(y_true, y_pred, eps_ = eps):
            return -(y_true*tf.log(y_pred + eps_) + (1-y_true)*tf.log((1-y_pred) + eps_))
        #r_y_true = tf.reshape(y_true, (y_true.shape[0], y_true.shape[1]//2, 2))
        loss_op = [tf.reduce_sum(self.target_pls[i][1] * my_binary_crossentropy(self.target_pls[i][0], self.prediction_ops[i]), axis=None, keep_dims=False) for i in range(0, self.batch_size)]
        loss_op = tf.add_n(loss_op, name=None) / self.batch_size #2
        return loss_op


    def train(self, sess, epochs, train_dataset, validation_dataset, output_dir, enable_plotting = 0, Targets_UnNormalization_fn = lambda x:x):
        '''

        Returns:

            dict with keys ['primary', 'secondacy'] mapping to validation scores (accuracy, auc OR rmse, r2)
        '''
        merged_summaries = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter(output_dir + '/train', sess.graph)
        #train_writer = tf.train.SummaryWriter(output_dir + '/train', sess.graph)
#        print('train_dataset',train_dataset)


        train_metric      = self.evaluate(sess, train_dataset, Targets_UnNormalization_fn=Targets_UnNormalization_fn)
        validation_metric = self.evaluate(sess, validation_dataset, Targets_UnNormalization_fn=Targets_UnNormalization_fn)
#        train_results_file_path = os.path.join(output_dir, "train_result.csv")

        if enable_plotting:
            plt.subplot(2, 1, 1)
            plt.title('Training data set')
            plt.axis([0, epochs, 0, train_metric['primary']])
            plt.subplot(2, 1, 2)
            plt.title('Vaidation data set')
            plt.axis([0, epochs, 0, validation_metric['primary']])
            plt.ion()
        logger.info('Start Training')

        steps_in_epoch = train_dataset.num_examples // self.batch_size
        # self.get_g_structure(sess,train_dataset)
        abort = 0
        for epoch in range(0, epochs):
            if abort:
                break
            for i in range(0, steps_in_epoch):
                feed_dict = self.fill_feed_dict(train_dataset, self.batch_size)
                none = sess.run([self.train_op], feed_dict=feed_dict)


            summ = sess.run(merged_summaries)
            train_writer.add_summary(summ, epoch)
            train_dataset.reset_epoch(permute=True)
            sess.run([self.global_step_update_op])

            if epoch % 5 == 0:
                train_metric = self.evaluate(sess, train_dataset, Targets_UnNormalization_fn=Targets_UnNormalization_fn)
                validation_metric = self.evaluate(sess, validation_dataset, Targets_UnNormalization_fn=Targets_UnNormalization_fn)
                if enable_plotting:
                    plt.subplot(2, 1, 1)
                    plt.scatter(epoch, train_metric['primary'], color='red', marker=".")
                    plt.scatter(epoch, train_metric['secondary'], color='blue', marker=".")
                    plt.subplot(2, 1, 2)
                    plt.scatter(epoch, validation_metric['primary'], color='red', marker=".")
                    plt.scatter(epoch, validation_metric['secondary'], color='blue', marker=".")
                    plt.pause(0.05)
                learning_rate = self.get_learning_rate(sess)

                if np.isnan(train_metric['primary']) and np.isnan(train_metric['secondary']):
                    logger.info("Epoch: {:}, Learning rate {:.8f},  All metrics are NaN: aborting training")
                    abort = 1
                    break

                if 'rmse' in train_metric:
                    logger.info(
                    "Epoch: {:}, Learning rate {:.8f},  Train RMSE: {:.4f}, Train R2: {:.4f}, Validation RMSE {:.4f}, Validation R2 {:.4f}".
                    format(epoch, learning_rate[0], train_metric['rmse'], train_metric['r2'],
                           validation_metric['rmse'], validation_metric['r2'], precision=8)
                    )
                else:
                    logger.info(
                    "Epoch: {:}, Learning rate {:.8f},  Train Accuracy {:.4f}, Train AUC: {:.4f}, Validation Accuracy {:.4f}, Validation AUC {:.4f}".
                    format(epoch, learning_rate[0], train_metric['accuracy'],
                           train_metric['auc'], validation_metric['accuracy'],
                           validation_metric['auc'], precision=8)
                    )
        training_predictions = Targets_UnNormalization_fn(self.predict(sess, train_dataset))
        save_results(output_dir, Targets_UnNormalization_fn(train_dataset.labels), training_predictions, additional_str='_train')
        validation_predictions = Targets_UnNormalization_fn(self.predict(sess, validation_dataset))
        save_results(output_dir, Targets_UnNormalization_fn(validation_dataset.labels), validation_predictions, additional_str='_valid')
        logger.info('Training Finished')
        return get_metric(training_predictions, Targets_UnNormalization_fn(train_dataset.labels)), get_metric(validation_predictions, Targets_UnNormalization_fn(validation_dataset.labels)) # python dict


    def evaluate(self, sess,  dataset, Targets_UnNormalization_fn = lambda x:x):
        predictions = self.predict(sess, dataset)
        targets = dataset.labels
        return get_metric(Targets_UnNormalization_fn(predictions), Targets_UnNormalization_fn(targets))

    def predict(self, sess, dataset):
        dataset.reset_epoch()
        predictions = []#np.empty(dataset.num_examples)
        for i in range(dataset.num_examples):
#            try:
            feed_dict = self.fill_feed_dict(dataset, 1)

            #print('i',i)
#            for k in feed_dict.keys():
#                try:
#                    ln = len(feed_dict[k])
#                except:
#                    ln = np.shape(feed_dict[k])
                #print ('  ', k, ln)
#            except:
#                print('predict:: skipped @',i,'of',dataset.num_examples)
#                continue
#                return predictions[:i]
            prediction_value = sess.run([self.prediction_ops[0]], feed_dict=feed_dict)
            predictions.append(prediction_value) #np.mean(prediction_value)

        predictions = np.squeeze(np.asarray(predictions))
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

        try:
            first_indices, second_indices = tf.split(value=indices, num_or_size_splits=2, axis=1) #v1.0 +
        except:
            first_indices, second_indices = tf.split(1, 2, indices) # before TF v1.0

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
    def apply_OutputNN(inputs, hidden_activation_type, output_activation_type):
        '''
        output_activation_type:

            e.g. use None (==linear) for regression problems

        '''
        hidden_activation_type = nn_utils.get_activation_fun(hidden_activation_type)
        output_activation_type = nn_utils.get_activation_fun(output_activation_type)

        with tf.variable_scope('hidden1') as scope:
            weights = tf.get_variable("weights")
            biases = tf.get_variable("biases")
            hidden1 = hidden_activation_type(tf.matmul(inputs, weights) + biases)


        with tf.variable_scope('output') as scope:
            weights = tf.get_variable("weights")
            out = tf.matmul(hidden1, weights)
            if output_activation_type is not None:
                return output_activation_type(out)
            return out




