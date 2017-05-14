#!/usr/bin/env bash

python -m InnerModel.train --batch_size=20 --smile_col='SMILES' --target_col='MTP' --training_file='ugrnn/data/karthikeyan/train_karthikeyan.csv' --validation_file='ugrnn/data/karthikeyan/validate_karthikeyan.csv' --model_name=model_0 --model_params 7,7,5  --output_dir=output/karthikeyan/ugrnn

python -m InnerModel.train --batch_size=20 --smile_col='SMILES' --target_col='MTP' --training_file='ugrnn/data/karthikeyan/train_karthikeyan.csv' --validation_file='ugrnn/data/karthikeyan/validate_karthikeyan.csv' --model_name=model_0 --model_params 7,7,5  --output_dir=output/karthikeyan/ugrnn-cr --contract_rings

python -m InnerModel.train --model_name=model_0 --batch_size=20 --smile_col='SMILES' --target_col='MTP' --training_file='ugrnn/data/karthikeyan/train_karthikeyan.csv' --validation_file='ugrnn/data/karthikeyan/validate_karthikeyan.csv' --model_params 7,7,5  --output_dir=output/karthikeyan/ugrnn-logp --add_logp

python -m InnerModel.train --model_name=model_0 --batch_size=20 --smile_col='SMILES' --target_col='MTP' --training_file='ugrnn/data/karthikeyan/train_karthikeyan.csv' --validation_file='ugrnn/data/karthikeyan/validate_karthikeyan.csv' --model_params 7,7,5  --output_dir=output/karthikeyan/ugrnn-cr-logp --contract_rings  --add_logp