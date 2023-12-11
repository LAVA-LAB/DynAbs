#!/bin/bash
# Shuttle benchmark against SReachTools
python3 RunFile.py --model_file JAIR22_models --model shuttle --timebound 16 --noise_samples 1600 --confidence 0.01 --prism_java_memory 8 --input_min -0.1 -0.1 --input_max 0.1 0.1 --partition_num_elem 20 10 4 4 --monte_carlo_iter 100 --x_init -0.75 -0.85 0.005 0.005 --plot
#
# 2D UAV benchmark
python3 RunFile.py --model_file JAIR22_models --model UAV --timebound 32 --UAV_dim 2 --noise_samples 3200 --confidence 0.01 --prism_java_memory 8 --input_min -4 -4 --input_max 4 4 --partition_num_elem 7 4 7 4 --plot
#
# 3D UAV benchmark
python3 RunFile.py --model_file JAIR22_models --model UAV --timebound 32 --UAV_dim 3 --noise_samples 6400 --noise_factor 0.1 --nongaussian_noise --input_min -4 -4 -4 --input_max 4 4 4 --partition_num_elem 15 3 9 3 7 3 --monte_carlo_iter 1000 --x_init -14 0 6 0 -2 0 --plot
python3 RunFile.py --model_file JAIR22_models --model UAV --timebound 32 --UAV_dim 3 --noise_samples 6400 --noise_factor 1 --nongaussian_noise --input_min -4 -4 -4 --input_max 4 4 4 --partition_num_elem 15 3 9 3 7 3 --monte_carlo_iter 1000 --x_init 14 0 6 0 -2 0 --plot
#
# 2D Building temperature control
python3 RunFile.py --model_file JAIR22_models --model building_2room --timebound 32 --noise_samples 3200 --confidence 0.01 --partition_num_elem 21 21 9 9 --plot
#
# 1D Building temperature control
python3 RunFile.py --model_file JAIR22_models --model building_1room --timebound 4 --noise_samples 3200 --confidence 0.01 --input_min 14 -10 --input_max 28 10 --partition_num_elem 19 20 --plot
python3 RunFile.py --model_file JAIR22_models --model building_1room --timebound 8 --noise_samples 3200 --confidence 0.01 --input_min 14 -10 --input_max 28 10 --partition_num_elem 19 20 --plot
python3 RunFile.py --model_file JAIR22_models --model building_1room --timebound 16 --noise_samples 3200 --confidence 0.01 --input_min 14 -10 --input_max 28 10 --partition_num_elem 19 20 --plot
python3 RunFile.py --model_file JAIR22_models --model building_1room --timebound 64 --noise_samples 3200 --confidence 0.01 --input_min 14 -10 --input_max 28 10 --partition_num_elem 19 20 --plot
python3 RunFile.py --model_file JAIR22_models --model building_1room --timebound inf --noise_samples 3200 --confidence 0.01 --input_min 14 -10 --input_max 28 10 --partition_num_elem 19 20 --plot
#
# Full spacecraft
python3 RunFile.py --model_file JAIR22_models --model spacecraft --timebound 32 --noise_samples 3200 --confidence 7.86e-9 --prism_java_memory 64 --input_min -2 -2 -2 --input_max 2 2 2 --partition_num_elem 11 23 5 5 5 5 --monte_carlo_iter 1000 --x_init 0.8 16 0 0 0 0 --plot
python3 RunFile.py --model_file JAIR22_models --model spacecraft --timebound 32 --noise_samples 3200 --confidence 7.86e-9 --prism_java_memory 64 --input_min -2 -2 -2 --input_max 2 2 2 --partition_num_elem 11 23 5 5 5 5 --monte_carlo_iter 1000 --x_init 0.8 16 0 0 0 0 --improved_synthesis --plot
python3 RunFile.py --model_file JAIR22_models --model spacecraft --timebound 32 --noise_samples 20000 --confidence 7.86e-9 --prism_java_memory 64 --input_min -2 -2 -2 --input_max 2 2 2 --partition_num_elem 11 23 5 5 5 5 --monte_carlo_iter 1000 --x_init 0.8 16 0 0 0 0 --plot
python3 RunFile.py --model_file JAIR22_models --model spacecraft --timebound 32 --noise_samples 20000 --confidence 7.86e-9 --prism_java_memory 64 --input_min -2 -2 -2 --input_max 2 2 2 --partition_num_elem 11 23 5 5 5 5 --monte_carlo_iter 1000 --x_init 0.8 16 0 0 0 0 --improved_synthesis --plot
#
# 2D spacecraft
python3 RunFile.py --model_file JAIR22_models --model spacecraft_2D --timebound 32 --noise_samples 3200 --confidence 0.01 --prism_java_memory 8 --input_min -2 -2 --input_max 2 2 --partition_num_elem 17 29 5 5 --monte_carlo_iter 1000 --x_init 1.2 19.9 0 0 --plot