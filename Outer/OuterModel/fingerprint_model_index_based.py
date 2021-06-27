"""
modifications to vanilla Keras:

added check_batch_dim argument to model.train_on_batch(...) and is save to model
made call to check_array_lengths() in train_on_batch() conditional on check_batch_dim


~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

relies on tf.gather/indexing to select features of neighboring atoms

"""

import keras.optimizers as optimizers
import keras.regularizers as regularizers
import keras.models as models
import keras.layers as layers
import keras.backend as backend
from . import config

try:
    import tensorflow as tf
    tf.python.control_flow_ops = tf
except:
    pass


def neural_fingerprint_layer(inputs, atom_features_of_previous_layer, num_atom_features,
                                 conv_width, fp_length, L2_reg, num_bond_features,
                                 batch_normalization = False, layer_index=0):
#    atom_features_of_previous_layer # either (variable_a, num_input_atom_features) [first layer] or (variable_a, conv_width)
    activations_by_degree = []
    for degree in config.ATOM_DEGREES:
        atom_features_of_previous_layer_this_degree = backend.sum(backend.gather(atom_features_of_previous_layer, indices=inputs['atom_neighbors_indices_degree_'+str(degree)]), 1)
        merged_atom_bond_features = layers.merge([atom_features_of_previous_layer_this_degree, inputs['bond_features_degree_'+str(degree)]], mode='concat', concat_axis=1)
        merged_atom_bond_features._keras_shape = (None, num_atom_features+num_bond_features)

        activations = layers.Dense(conv_width, activation='relu', bias=False, name='activations_{}_degree_{}'.format(layer_index, degree))(merged_atom_bond_features)

        activations_by_degree.append(activations)
    # skip-connection to output/final fingerprint
    output_to_fingerprint_tmp = layers.Dense(fp_length, activation='softmax', name = 'fingerprint_skip_connection_{}'.format(layer_index))(atom_features_of_previous_layer) # (variable_a, fp_length)
    output_to_fingerprint     = layers.Lambda(lambda x: backend.dot(inputs['atom_batch_matching_matrix_degree_'+str(degree)], x))(output_to_fingerprint_tmp)  # layers.Lambda(lambda x: backend.dot(inputs['atom_batch_matching_matrix_degree_'+str(degree)], x))(output_to_fingerprint_tmp) # (batch_size, fp_length)
    # connect to next layer
    this_activations_tmp = layers.Dense(conv_width, activation='relu', name='layer_{}_activations'.format(layer_index))(atom_features_of_previous_layer) # (variable_a, conv_width)
    # (variable_a, conv_width)
    merged_neighbor_activations = layers.merge(activations_by_degree, mode='concat',concat_axis=0)
    new_atom_features = layers.Lambda(lambda x:merged_neighbor_activations + x)(this_activations_tmp ) #(variable_a, conv_width)
    if batch_normalization:
        new_atom_features = layers.normalization.BatchNormalization()(new_atom_features)
    return new_atom_features, output_to_fingerprint


def build_fingerprint_model(fp_length = 50, fp_depth = 4, conv_width = 20,
                                             predictor_MLP_layers = [200, 200, 200],
                                             L2_reg = 3e-4, num_input_atom_features = 62,
                                             num_bond_features = 6, batch_normalization = False,
                                             regression = True, number_of_classes = 2,
                                             binary_multitask = False,
                                             masked_loss_function = False):
    """
    fp_length:

    	Size of the fingerprint vector, which is the collection of contributions of all convolutional layers and is the input to the multilayer perceptron (controlled by predictor_MLP_layers)

    fp_depth:

    	The depth of the convolutional network (i.e. number of layers) - this determines the effective size of the computed features

    conv_width:

    	Size of the hidden vectors used in the convolutional layers

    predictor_MLP_layers:

    	List of integer values, each number selects the number of neurons in a fully connected layer (i.e. [200, 200, 200] will create an MLP with three layers with 200 neurons each)

    L2_reg:

    	Strength of L2 weight decay for parameter regularization

    regression (binary):

        Select whether the data set is a regression (True) or classification (False) task.

    number_of_classes:

        Number of classes in data set (if classification task)

    binary_multitask:

        set to True for multitask binary prediction problems (e.g. Toxcast or Tox21); uses <number_of_classes> many sigmoid output units and trains the network using binary crossentropy loss.

    masked_loss_function:

    	One output of:

    	masked_labels, masked_loss_function = OuterModel.train_helper.create_labels_NaN_mask(labels)

        If True: compiled model will expect that labels are a tuple of (classes, binary_mask), where the values of binary_mask should be set to 0 at all positions/classes of the batch that are to be ignored and 1 for the rest.
        Everything is automatically handled by the implementation, provided that the create_labels_NaN_mask() function was used beforehand.
    """
    if masked_loss_function:
        if regression or not binary_multitask:
            raise NotImplementedError('masked_loss_function currently only implemented for binary_multitask classification')
    inputs = {}
    inputs['input_atom_features'] = layers.Input(name='input_atom_features', shape=(num_input_atom_features,))
    for degree in config.ATOM_DEGREES:
        inputs['bond_features_degree_'+str(degree)] = layers.Input(name='bond_features_degree_'+str(degree),
                                                            shape=(num_bond_features,))
        inputs['atom_neighbors_indices_degree_'+str(degree)] = layers.Input(name='atom_neighbors_indices_degree_'+str(degree), shape=(degree,), dtype = 'int32') #todo shape

#        inputs['input_atom_features_degree_'+str(degree)] = layers.Input(name='input_atom_features_degree_'+str(degree), shape=(num_input_atom_features,))

        inputs['atom_batch_matching_matrix_degree_'+str(degree)] = layers.Input(name='atom_batch_matching_matrix_degree_'+str(degree), shape=(None,)) # shape is (batch_size, variable_a)
    atom_features = inputs['input_atom_features']
    all_outputs_to_fingerprint = []
    num_atom_features = num_input_atom_features
    for i in range(fp_depth):
        atom_features, output_to_fingerprint = neural_fingerprint_layer(inputs, atom_features_of_previous_layer = atom_features,
                                                                        num_atom_features = num_atom_features, conv_width = conv_width,
                                                                        fp_length = fp_length, L2_reg = L2_reg,
                                                                        num_bond_features = num_bond_features,
                                                                        batch_normalization = batch_normalization,
                                                                        layer_index=i)
        num_atom_features = conv_width
        all_outputs_to_fingerprint.append(output_to_fingerprint)
    # THIS is the actual fingerprint, we will feed it into an MLP for prediction  -- shape is (batch_size, fp_length)
    neural_fingerprint = layers.merge(all_outputs_to_fingerprint, mode='sum') if len(all_outputs_to_fingerprint)>1 else all_outputs_to_fingerprint
    Prediction_MLP_layer = neural_fingerprint
    for i, hidden in enumerate(predictor_MLP_layers):
        Prediction_MLP_layer = layers.Dense(hidden, activation='relu', W_regularizer=regularizers.l2(L2_reg), name='MLP_hidden_'+str(i))(Prediction_MLP_layer)

    if regression:
        main_prediction = layers.Dense(1, activation='linear', name='main_prediction')(Prediction_MLP_layer)
        model = models.Model(input=list(inputs.values()), output=[main_prediction])
        model.compile(optimizer=optimizers.Adam(), loss={'main_prediction':'mse'})
    else:
        assert number_of_classes>1, 'invalid <number_of_classes>'
        main_prediction = layers.Dense(number_of_classes, activation=('sigmoid' if binary_multitask else 'softmax'), name='main_prediction')(Prediction_MLP_layer)
        model = models.Model(input=list(inputs.values()), output=[main_prediction])
        if masked_loss_function:
            def cross_ent(real, pred, eps = 1e-6):
                return -(real*backend.log(pred + eps) + (1-real)*backend.log((1-pred) + eps))

            def my_binary_crossentropy(y_true, y_pred): #masked
                r_y_true = backend.reshape(y_true, (backend.shape(y_true)[0], backend.shape(y_true)[1]//2, 2))
                return backend.mean(r_y_true[:,:,1] * cross_ent(r_y_true[:,:,0], y_pred), axis=-1)
        else:
            my_binary_crossentropy = 'binary_crossentropy'
        model.compile(optimizer=optimizers.Adam(), loss={'main_prediction': my_binary_crossentropy if binary_multitask else 'categorical_crossentropy'})
    return model
