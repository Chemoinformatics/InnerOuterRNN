The Outer Approach in Chemoinformatics (Keras-based implementation of Neural Fingerprints)
=================================================

This is a Keras (Tensorflow & Theano) implementation based on the model proposed in [Convolutional Networks on Graphs for Learning Molecular Fingerprints (Duvenaud et al.)](http://arxiv.org/pdf/1509.09292.pdf)

It implements convolutional neural networks operating on molecular graphs (SMILES) of arbitrary size for chemical property prediction (e.g. solubility).

### Short summary of the method: 

The model consists of a sequence of "convolutional" layers which accumulate, for each atom in the molecule, the bond- and atom features of directly neighboring atoms. These accumulated features are used together with the layer's learnable set of weights to compute a new set of atom features for the next convolutional layer in the model. Each convolutional layer's newly updated/computed set of atom features is used to compute a contribution to the overall neural fingerprint vector, and accumulation is done via summation of individual contributions. The graph that represents this model is "orthogonal" to the molecular graph, thus we call this an "outer" approach to differentiate it from the "inner" approach, which consists of a recurrent network that crawls the moclecular graph itself instead of stacking convolutional layers on it.

## Requirements:

Python, Numpy -- preferrably using [Anaconda](https://www.continuum.io/downloads)

Either [Theano](http://deeplearning.net/software/theano/install.html) or [Tensorflow](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html)

[RDkit](http://www.rdkit.org/docs/Install.html) -- the easiest way to install it when using Anaconda is "conda install -c https://conda.anaconda.org/rdkit rdkit"


---------------------------------------


The original implementation using numpy/autograd can be found at (https://github.com/HIPS/neural-fingerprint)