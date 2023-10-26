#!/bin/bash
# Shuttle benchmark against SReachTools
python3 RunFile.py --model_file JAIR22_models --model shuttle --timebound 16 --noise_samples 1600 --confidence 0.01 --prism_java_memory 8 --monte_carlo_iter 100 --x_init '[-0.75, -0.85, 0.005, 0.005]' --plot
#
# 2D UAV benchmark
python3 RunFile.py --model_file JAIR22_models --model UAV --timebound 32 --UAV_dim 2 --noise_samples 3200 --confidence 0.01 --prism_java_memory 8 --plot
#
# 3D UAV benchmark
python3 RunFile.py --model_file JAIR22_models --model UAV --timebound 32 --UAV_dim 3 --noise_samples 6400 --prism_java_memory 14 --noise_factor 0.1 --nongaussian_noise --monte_carlo_iter 1000 --x_init '[-14,0,6,0,-2,0]' --plot
python3 RunFile.py --model_file JAIR22_models --model UAV --timebound 32 --UAV_dim 3 --noise_samples 6400 --prism_java_memory 14 --noise_factor 1 --nongaussian_noise --monte_carlo_iter 1000 --x_init '[-14,0,6,0,-2,0]' --plot
#
# 2D Building temperature control
python3 RunFile.py --model_file JAIR22_models --model building_2room --timebound 32 --noise_samples 3200 --confidence 0.01 --prism_java_memory 14 --plot
#
# 1D Building temperature control
python3 RunFile.py --model_file JAIR22_models --model building_1room --timebound 4 --noise_samples 3200 --confidence 0.01 --plot
python3 RunFile.py --model_file JAIR22_models --model building_1room --timebound 8 --noise_samples 3200 --confidence 0.01 --plot
python3 RunFile.py --model_file JAIR22_models --model building_1room --timebound 16 --noise_samples 3200 --confidence 0.01 --plot
python3 RunFile.py --model_file JAIR22_models --model building_1room --timebound 64 --noise_samples 3200 --confidence 0.01 --plot
python3 RunFile.py --model_file JAIR22_models --model building_1room --timebound inf --noise_samples 3200 --confidence 0.01 --plot
#
# Full spacecraft
python3 RunFile.py --model_file JAIR22_models --model spacecraft --timebound 32 --noise_samples 3200 --confidence 7.86e-9 --prism_java_memory 64 --monte_carlo_iter 1000 --x_init '[0.8, 16, 0, 0, 0, 0]' --plot
python3 RunFile.py --model_file JAIR22_models --model spacecraft --timebound 32 --noise_samples 3200 --confidence 7.86e-9 --prism_java_memory 64 --monte_carlo_iter 1000 --x_init '[0.8, 16, 0, 0, 0, 0]' --improved_synthesis --plot
python3 RunFile.py --model_file JAIR22_models --model spacecraft --timebound 32 --noise_samples 20000 --confidence 7.86e-9 --prism_java_memory 64 --monte_carlo_iter 1000 --x_init '[0.8, 16, 0, 0, 0, 0]' --plot
python3 RunFile.py --model_file JAIR22_models --model spacecraft --timebound 32 --noise_samples 20000 --confidence 7.86e-9 --prism_java_memory 64 --monte_carlo_iter 1000 --x_init '[0.8, 16, 0, 0, 0, 0]' --improved_synthesis --plot
#
# 2D spacecraft
python3 RunFile.py --model_file JAIR22_models --model spacecraft_2D --timebound 32 --noise_samples 3200 --confidence 0.01 --prism_java_memory 8 --monte_carlo_iter 1000 --x_init '[1.2, 19.9, 0, 0]' --plot