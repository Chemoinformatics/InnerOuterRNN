This is a modified version of keras!

- Added check_batch_dim argument to model.train_on_batch(...) and made call to check_array_lengths() in train_on_batch() conditional on check_batch_dim

The reason for this is to support inputs with varying batch-sizes which is needed for implementing neural-fingerprints. The alternative would be to use tensor-reshape operations (which is less efficient).