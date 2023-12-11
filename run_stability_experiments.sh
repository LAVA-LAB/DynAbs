#!/bin/bash
for ii in 50; do
  for jj in 50 40 30 20 10; do
	    python3 RunFile.py --model_file JAIR22_models --model robot --timebound 16 --noise_samples 1600 --confidence 0.01 --prism_java_memory 8 --input_min -$ii --input_max $ii --input_min_constr -$jj --input_max_constr $jj --partition_num_elem 41 41 --plot --model_params '{"stabilizing_controller": "True", "stability_param": 0.5}';
  done
done

for ii in 50 40 30 20 10; do
  python3 RunFile.py --model_file JAIR22_models --model robot --timebound 16 --noise_samples 1600 --confidence 0.01 --prism_java_memory 8 --input_min -$ii --input_max $ii --partition_num_elem 41 41 --plot --model_params '{"stabilizing_controller": "False", "stability_param": 0.5}';
done

python3 RunFile.py --model_file JAIR22_models --model robot --timebound 16 --noise_samples 1600 --confidence 0.01 --prism_java_memory 8 --input_min -25 --input_max 25 --partition_num_elem 41 41 --plot --model_params '{"stabilizing_controller": "False", "stability_param": 0.5}';