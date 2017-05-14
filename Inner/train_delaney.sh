#!/usr/bin/env bash

python -m InnerModel.train --model_name=model_0 --model_params 7,3,5  --output_dir=output/delaney/ugrnn
python -m InnerModel.train --model_name=model_1 --model_params 7,4,5  --output_dir=output/delaney/ugrnn
python -m InnerModel.train --model_name=model_2 --model_params 7,5,5  --output_dir=output/delaney/ugrnn
python -m InnerModel.train --model_name=model_3 --model_params 7,6,5  --output_dir=output/delaney/ugrnn
python -m InnerModel.train --model_name=model_4 --model_params 7,7,5  --output_dir=output/delaney/ugrnn
python -m InnerModel.train --model_name=model_5 --model_params 7,8,5  --output_dir=output/delaney/ugrnn
python -m InnerModel.train --model_name=model_6 --model_params 7,9,5  --output_dir=output/delaney/ugrnn
python -m InnerModel.train --model_name=model_7 --model_params 7,10,5  --output_dir=output/delaney/ugrnn
python -m InnerModel.train --model_name=model_8 --model_params 7,11,5  --output_dir=output/delaney/ugrnn
python -m InnerModel.train --model_name=model_9 --model_params 7,12,5  --output_dir=output/delaney/ugrnn
python -m InnerModel.train --model_name=model_10 --model_params 3,3,5  --output_dir=output/delaney/ugrnn
python -m InnerModel.train --model_name=model_11 --model_params 4,3,5  --output_dir=output/delaney/ugrnn
python -m InnerModel.train --model_name=model_12 --model_params 5,3,5  --output_dir=output/delaney/ugrnn
python -m InnerModel.train --model_name=model_13 --model_params 6,3,5  --output_dir=output/delaney/ugrnn
python -m InnerModel.train --model_name=model_14 --model_params 7,3,5  --output_dir=output/delaney/ugrnn
python -m InnerModel.train --model_name=model_15 --model_params 8,3,5  --output_dir=output/delaney/ugrnn
python -m InnerModel.train --model_name=model_16 --model_params 9,3,5  --output_dir=output/delaney/ugrnn
python -m InnerModel.train --model_name=model_17 --model_params 10,3,5  --output_dir=output/delaney/ugrnn
python -m InnerModel.train --model_name=model_18 --model_params 11,3,5  --output_dir=output/delaney/ugrnn
python -m InnerModel.train --model_name=model_19 --model_params 12,3,5  --output_dir=output/delaney/ugrnn


python -m InnerModel.train --model_name=model_0 --model_params 7,3,5  --output_dir=output/delaney/ugrnn-cr --contract_rings
python -m InnerModel.train --model_name=model_1 --model_params 7,4,5  --output_dir=output/delaney/ugrnn-cr --contract_rings
python -m InnerModel.train --model_name=model_2 --model_params 7,5,5  --output_dir=output/delaney/ugrnn-cr --contract_rings
python -m InnerModel.train --model_name=model_3 --model_params 7,6,5  --output_dir=output/delaney/ugrnn-cr --contract_rings
python -m InnerModel.train --model_name=model_4 --model_params 7,7,5  --output_dir=output/delaney/ugrnn-cr --contract_rings
python -m InnerModel.train --model_name=model_5 --model_params 7,8,5  --output_dir=output/delaney/ugrnn-cr --contract_rings
python -m InnerModel.train --model_name=model_6 --model_params 7,9,5  --output_dir=output/delaney/ugrnn-cr --contract_rings
python -m InnerModel.train --model_name=model_7 --model_params 7,10,5  --output_dir=output/delaney/ugrnn-cr --contract_rings
python -m InnerModel.train --model_name=model_8 --model_params 7,11,5  --output_dir=output/delaney/ugrnn-cr --contract_rings
python -m InnerModel.train --model_name=model_9 --model_params 7,12,5  --output_dir=output/delaney/ugrnn-cr --contract_rings
python -m InnerModel.train --model_name=model_10 --model_params 3,3,5  --output_dir=output/delaney/ugrnn-cr --contract_rings
python -m InnerModel.train --model_name=model_11 --model_params 4,3,5  --output_dir=output/delaney/ugrnn-cr --contract_rings
python -m InnerModel.train --model_name=model_12 --model_params 5,3,5  --output_dir=output/delaney/ugrnn-cr --contract_rings
python -m InnerModel.train --model_name=model_13 --model_params 6,3,5  --output_dir=output/delaney/ugrnn-cr --contract_rings
python -m InnerModel.train --model_name=model_14 --model_params 7,3,5  --output_dir=output/delaney/ugrnn-cr --contract_rings
python -m InnerModel.train --model_name=model_15 --model_params 8,3,5  --output_dir=output/delaney/ugrnn-cr --contract_rings
python -m InnerModel.train --model_name=model_16 --model_params 9,3,5  --output_dir=output/delaney/ugrnn-cr --contract_rings
python -m InnerModel.train --model_name=model_17 --model_params 10,3,5  --output_dir=output/delaney/ugrnn-cr --contract_rings
python -m InnerModel.train --model_name=model_18 --model_params 11,3,5  --output_dir=output/delaney/ugrnn-cr --contract_rings
python -m InnerModel.train --model_name=model_19 --model_params 12,3,5  --output_dir=output/delaney/ugrnn-cr --contract_rings

python -m InnerModel.train --model_name=model_0 --model_params 7,3,5  --output_dir=output/delaney/ugrnn-logp --add_logp
python -m InnerModel.train --model_name=model_1 --model_params 7,4,5  --output_dir=output/delaney/ugrnn-logp --add_logp
python -m InnerModel.train --model_name=model_2 --model_params 7,5,5  --output_dir=output/delaney/ugrnn-logp --add_logp
python -m InnerModel.train --model_name=model_3 --model_params 7,6,5  --output_dir=output/delaney/ugrnn-logp --add_logp
python -m InnerModel.train --model_name=model_4 --model_params 7,7,5  --output_dir=output/delaney/ugrnn-logp --add_logp
python -m InnerModel.train --model_name=model_5 --model_params 7,8,5  --output_dir=output/delaney/ugrnn-logp --add_logp
python -m InnerModel.train --model_name=model_6 --model_params 7,9,5  --output_dir=output/delaney/ugrnn-logp --add_logp
python -m InnerModel.train --model_name=model_7 --model_params 7,10,5  --output_dir=output/delaney/ugrnn-logp --add_logp
python -m InnerModel.train --model_name=model_8 --model_params 7,11,5  --output_dir=output/delaney/ugrnn-logp --add_logp
python -m InnerModel.train --model_name=model_9 --model_params 7,12,5  --output_dir=output/delaney/ugrnn-logp --add_logp
python -m InnerModel.train --model_name=model_10 --model_params 3,3,5  --output_dir=output/delaney/ugrnn-logp --add_logp
python -m InnerModel.train --model_name=model_11 --model_params 4,3,5  --output_dir=output/delaney/ugrnn-logp --add_logp
python -m InnerModel.train --model_name=model_12 --model_params 5,3,5  --output_dir=output/delaney/ugrnn-logp --add_logp
python -m InnerModel.train --model_name=model_13 --model_params 6,3,5  --output_dir=output/delaney/ugrnn-logp --add_logp
python -m InnerModel.train --model_name=model_14 --model_params 7,3,5  --output_dir=output/delaney/ugrnn-logp --add_logp
python -m InnerModel.train --model_name=model_15 --model_params 8,3,5  --output_dir=output/delaney/ugrnn-logp --add_logp
python -m InnerModel.train --model_name=model_16 --model_params 9,3,5  --output_dir=output/delaney/ugrnn-logp --add_logp
python -m InnerModel.train --model_name=model_17 --model_params 10,3,5  --output_dir=output/delaney/ugrnn-logp --add_logp
python -m InnerModel.train --model_name=model_18 --model_params 11,3,5  --output_dir=output/delaney/ugrnn-logp --add_logp
python -m InnerModel.train --model_name=model_19 --model_params 12,3,5  --output_dir=output/delaney/ugrnn-logp --add_logp


python -m InnerModel.train --model_name=model_0 --model_params 7,3,5  --output_dir=output/delaney/ugrnn-cr-logp --contract_rings  --add_logp
python -m InnerModel.train --model_name=model_1 --model_params 7,4,5  --output_dir=output/delaney/ugrnn-cr-logp --contract_rings --add_logp
python -m InnerModel.train --model_name=model_2 --model_params 7,5,5  --output_dir=output/delaney/ugrnn-cr-logp --contract_rings --add_logp
python -m InnerModel.train --model_name=model_3 --model_params 7,6,5  --output_dir=output/delaney/ugrnn-cr-logp --contract_rings --add_logp
python -m InnerModel.train --model_name=model_4 --model_params 7,7,5  --output_dir=output/delaney/ugrnn-cr-logp --contract_rings --add_logp
python -m InnerModel.train --model_name=model_5 --model_params 7,8,5  --output_dir=output/delaney/ugrnn-cr-logp --contract_rings --add_logp
python -m InnerModel.train --model_name=model_6 --model_params 7,9,5  --output_dir=output/delaney/ugrnn-cr-logp --contract_rings --add_logp
python -m InnerModel.train --model_name=model_7 --model_params 7,10,5  --output_dir=output/delaney/ugrnn-cr-logp --contract_rings --add_logp
python -m InnerModel.train --model_name=model_8 --model_params 7,11,5  --output_dir=output/delaney/ugrnn-cr-logp --contract_rings --add_logp
python -m InnerModel.train --model_name=model_9 --model_params 7,12,5  --output_dir=output/delaney/ugrnn-cr-logp --contract_rings --add_logp
python -m InnerModel.train --model_name=model_10 --model_params 3,3,5  --output_dir=output/delaney/ugrnn-cr-logp --contract_rings --add_logp
python -m InnerModel.train --model_name=model_11 --model_params 4,3,5  --output_dir=output/delaney/ugrnn-cr-logp --contract_rings --add_logp
python -m InnerModel.train --model_name=model_12 --model_params 5,3,5  --output_dir=output/delaney/ugrnn-cr-logp --contract_rings --add_logp
python -m InnerModel.train --model_name=model_13 --model_params 6,3,5  --output_dir=output/delaney/ugrnn-cr-logp --contract_rings --add_logp
python -m InnerModel.train --model_name=model_14 --model_params 7,3,5  --output_dir=output/delaney/ugrnn-cr-logp --contract_rings --add_logp
python -m InnerModel.train --model_name=model_15 --model_params 8,3,5  --output_dir=output/delaney/ugrnn-cr-logp --contract_rings --add_logp
python -m InnerModel.train --model_name=model_16 --model_params 9,3,5  --output_dir=output/delaney/ugrnn-cr-logp --contract_rings --add_logp
python -m InnerModel.train --model_name=model_17 --model_params 10,3,5  --output_dir=output/delaney/ugrnn-cr-logp --contract_rings --add_logp
python -m InnerModel.train --model_name=model_18 --model_params 11,3,5  --output_dir=output/delaney/ugrnn-cr-logp --contract_rings --add_logp
python -m InnerModel.train --model_name=model_19 --model_params 12,3,5  --output_dir=output/delaney/ugrnn-cr-logp --contract_rings --add_logp
