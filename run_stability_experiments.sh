#!/bin/bash
for ii in 0.125 0.25 0.5 1 2 4 8; do
  for jj in 0.125 0.25 0.5 1 2 4 8; do
  	echo $ii;
  	echo $jj;
	python3 RunFile.py --model_file JAIR22_models --model robot --timebound 16 --noise_samples 1600 --confidence 0.01 --prism_java_memory 8 --plot --model_params '{"u_multiply": '$ii', "stability_param": '$jj'}';
  done
done

python3 RunFile.py --model_file JAIR22_models --model robot --timebound 16 --noise_samples 1600 --confidence 0.01 --prism_java_memory 8 --plot --model_params '{"u_multiply": 1, "stability_param": 0.5}';
