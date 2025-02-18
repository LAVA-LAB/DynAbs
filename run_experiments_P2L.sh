#!/bin/bash
#prism_exe='/Users/thobad/Documents/Tools/prism/prism/bin/prism'
prism_exe='/home/tbadings/Tools/prism/prism/bin/prism'
#
# 3D UAV benchmark

all_flags="--model_file JAIR22_models --model UAV --timebound 32 --UAV_dim 3 --prism_executable $prism_exe --noise_samples $N --prism_java_memory 14 --noise_factor 1 --nongaussian_noise --monte_carlo_iter 100000 --x_init '[-14,0,6,0,-2,0]'"
p2l_flags="--P2L --P2L_add_per_iteration 50 --P2L_pretrain_fraction 0.5 --P2L_delta 0.0082664" # This delta/confidence corresponds with a confidence of 1e-8 on the individual probability intervals of the IMDP

for seed in 1
do
  for N in 160 320 1280 6400
  do
    python3 RunFile.py $all_flags --clean_prism_model --seed $seed;
    python3 RunFile.py $all_flags --clean_prism_model --seed $seed --clopper_pearson;
    python3 RunFile.py $all_flags --clean_prism_model --seed $seed --mdp_mode estimate;
  done
  for N in 160 320 1280 6400
  do
    python3 RunFile.py $all_flags --clean_prism_model --seed $seed --p2l_flags;
    python3 RunFile.py $all_flags --clean_prism_model --seed $seed --p2l_flags --clopper_pearson;
    python3 RunFile.py $all_flags --clean_prism_model --seed $seed --p2l_flags --mdp_mode estimate;
  done
done