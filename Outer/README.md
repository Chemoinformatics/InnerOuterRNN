The Outer Approach in Chemoinformatics
=================================================

This is a Keras (Tensorflow & Theano) implementation based on the model proposed in [Convolutional Networks on Graphs for Learning Molecular Fingerprints (Duvenaud et al.)](http://arxiv.org/pdf/1509.09292.pdf)

The implementation allows the training of convolutional neural networks operating on molecular graphs (SMILES) of arbitrary size for predicting chemical/physical or biological properties of small molecules. It can handle regression, classification, and multitask binary-classification problems with or without missing entries in the labels (e.g. such as in the case of the Toxcast and Tox21 data sets).

### Short summary of the method: 

The model consists of a sequence of "convolutional" layers which accumulate, for each atom in the molecule, the bond- and atom features of directly neighboring atoms. These accumulated features are used together with the layer's learnable set of weights to compute a new set of atom features for the next convolutional layer in the model. Each convolutional layer's newly updated/computed set of atom features is used to compute a contribution to the overall neural fingerprint vector, and accumulation is done via summation of individual contributions. The graph that represents this model is "orthogonal" to the molecular graph, thus we call this an "outer" approach to differentiate it from the "inner" approach, which consists of a recurrent network that crawls the moclecular graph itself instead of stacking convolutional layers on it.


## Requirements:


Python 2 or 3, Numpy. We suggest using [Anaconda](https://www.continuum.io/downloads)

Either [Theano](http://deeplearning.net/software/theano/install.html) or [Tensorflow](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html)
A GPU with CUDA installation is not required to use the code, but could in some cases offer a small speed advantage.


[RDkit](http://www.rdkit.org/docs/Install.html) -- the easiest way to install it when using Anaconda is "conda install -c https://conda.anaconda.org/rdkit rdkit"




## Usage:

Several code examples are given in the examples/ subdirectory, which can be used as a starting point for new data sets or tasks. Example files with the prefix "SingleModelTraining" are probably most useful for this purpose, while examples starting with the prefix "CV" perform a 10-fold cross-validation on the respective data set on a fairly abstract level.

### Loading data:

The implementation expects the input data to be stored as .csv file, with one column containing the SMILES and at least a subset of the others containing the prediction targets. The data can be loaded using the train\_helper.load\_and\_cache\_csv() function, which will detect and remove duplicate samples, and those that RDKit detects as "invalid", from the data set. The processed data is the (optionally) cached.

### Building the Model:

The regression or classification model is constructed by calling the following functing with appropriate parameters:

	model = OuterModel.fingerprint_model_index_based.build_fingerprint_model(fp_length=50, fp_depth=4, conv_width=20, predictor_MLP_layers=[200, 200, 200], L2_reg=3e-4, regression=True, number_of_classes=2, binary_multitask=False, masked_loss_function=False)

fp\_length:

	Size of the fingerprint vector, which is the collection of contributions of all convolutional layers and is the input to the multilayer perceptron (controlled by predictor_MLP_layers)

fp\_depth:

	The depth of the convolutional network (i.e. number of layers) - this determines the effective size of the computed features

conv\_width:

	Size of the hidden vectors used in the convolutional layers

predictor\_MLP\_layers:

	List of integer values, each number selects the number of neurons in a fully connected layer (i.e. [200, 200, 200] will create an MLP with three layers with 200 neurons each)

L2\_reg:

	Strength of L2 weight decay for parameter regularization

regression (binary): 
    
    Select whether the data set is a regression (True) or classification (False) task.
    
number\_of\_classes: 
    
    Number of classes in data set (if classification task)

binary\_multitask:
    
    set to True for multitask binary prediction problems (e.g. Toxcast or Tox21); uses <number_of_classes> many sigmoid output units and trains the network using binary crossentropy loss.

masked\_loss\_function:
    
	One output of:

	masked_labels, masked_loss_function = OuterModel.train_helper.create_labels_NaN_mask(labels)

    If True: compiled model will expect that labels are a tuple of (classes, binary_mask), where the values of binary_mask should be set to 0 at all positions/classes of the batch that are to be ignored and 1 for the rest. Everything is automatically handled by the implementation, provided that the create_labels_NaN_mask() function was used beforehand.


### Training the Model:

Use the following function to transform the data from a list of SMILES into a form that can be used by the compiled Keras model (it will return a list of dictionaries that encode features of the molecular graphs, and uses RDKit for this purpose):

	train_data, validation_data, test_data = data_preprocessing.preprocess_data_set_for_Model(train_data, validation_data, test_data, training_batchsize = 20, testset_batchsize = 1000)

The following function will train the model for a given number of epochs, starting with a given initial learning rate which will be decayed to <total_lr_decay> of its starting value over the course of training:

	OuterModel.train_outer.train_model(model, train_data, valid_data, test_data, initial_lr=0.002, total_lr_decay=0.01, batchsize = 20, num_epochs = 120, regression = False)

the training function returns the model and train/validation/test scores of the model at the point where it reached the best validation score (early stopping).



---------------------------------------


The original implementation using numpy/autograd can be found at (https://github.com/HIPS/neural-fingerprint)