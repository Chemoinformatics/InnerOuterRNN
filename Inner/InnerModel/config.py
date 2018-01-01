learning_rate_decay_factor = .98 # learning rate is multiplied with this factor after each epoch during training
max_seq_len = 180 # maximum allowed SMILES length, larger ones will be removed before training during data preprocessing. Increasing this has an almost quadratic negative impact on run-time.
activation_function="relu"
initializer="xavier"

