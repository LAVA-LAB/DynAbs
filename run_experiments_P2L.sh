#!/bin/bash
#prism_exe='/Users/thobad/Documents/Tools/prism/prism/bin/prism'
prism_exe='/home/tbadings/Tools/prism/prism/bin/prism'
#
# 3D UAV benchmark
for seed in 1 2 3 4 5
do
  for N in 320 1280 6400
  do
    python3 RunFile.py --clean_prism_model --model_file JAIR22_models --model UAV --timebound 32 --UAV_dim 3 --prism_executable $prism_exe --noise_samples $N --prism_java_memory 14 --noise_factor 1 --nongaussian_noise --monte_carlo_iter 10000 --x_init '[-14,0,6,0,-2,0]' --seed $seed
    python3 RunFile.py --clean_prism_model --mdp_mode estimate --model_file JAIR22_models --model UAV --timebound 32 --UAV_dim 3 --prism_executable $prism_exe --noise_samples $N --prism_java_memory 14 --noise_factor 1 --nongaussian_noise --monte_carlo_iter 10000 --x_init '[-14,0,6,0,-2,0]' --seed $seed
  #
    python3 RunFile.py --clean_prism_model --clopper_pearson --model_file JAIR22_models --model UAV --timebound 32 --UAV_dim 3 --prism_executable $prism_exe --noise_samples $N --prism_java_memory 14 --noise_factor 1 --nongaussian_noise --monte_carlo_iter 10000 --x_init '[-14,0,6,0,-2,0]' --seed $seed
    python3 RunFile.py --clean_prism_model --clopper_pearson --mdp_mode estimate --model_file JAIR22_models --model UAV --timebound 32 --UAV_dim 3 --prism_executable $prism_exe --noise_samples $N --prism_java_memory 14 --noise_factor 1 --nongaussian_noise --monte_carlo_iter 10000 --x_init '[-14,0,6,0,-2,0]' --seed $seed
  done
  for N in 320 1280 6400
  do
    python3 RunFile.py --clean_prism_model --model_file JAIR22_models --model UAV --timebound 32 --UAV_dim 3 --prism_executable $prism_exe --noise_samples $N --prism_java_memory 14 --noise_factor 1 --nongaussian_noise --monte_carlo_iter 10000 --x_init '[-14,0,6,0,-2,0]' --seed $seed --P2L --P2L_add_per_iteration 50 --P2L_pretrain_fraction 0.5 --P2L_delta 0.0082664
    python3 RunFile.py --clean_prism_model --mdp_mode estimate --model_file JAIR22_models --model UAV --timebound 32 --UAV_dim 3 --prism_executable $prism_exe --noise_samples $N --prism_java_memory 14 --noise_factor 1 --nongaussian_noise --monte_carlo_iter 10000 --x_init '[-14,0,6,0,-2,0]' --seed $seed --P2L --P2L_add_per_iteration 50 --P2L_pretrain_fraction 0.5 --P2L_delta 0.0082664
    ##
    python3 RunFile.py --clean_prism_model --clopper_pearson --model_file JAIR22_models --model UAV --timebound 32 --UAV_dim 3 --prism_executable $prism_exe --noise_samples $N --prism_java_memory 14 --noise_factor 1 --nongaussian_noise --monte_carlo_iter 10000 --x_init '[-14,0,6,0,-2,0]' --seed $seed --P2L --P2L_add_per_iteration 50 --P2L_pretrain_fraction 0.5 --P2L_delta 0.0082664
    python3 RunFile.py --clean_prism_model --clopper_pearson --mdp_mode estimate --model_file JAIR22_models --model UAV --timebound 32 --UAV_dim 3 --prism_executable $prism_exe --noise_samples $N --prism_java_memory 14 --noise_factor 1 --nongaussian_noise --monte_carlo_iter 10000 --x_init '[-14,0,6,0,-2,0]' --seed $seed --P2L --P2L_add_per_iteration 50 --P2L_pretrain_fraction 0.5 --P2L_delta 0.0082664
  done
done