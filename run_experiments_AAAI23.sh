#!/bin/bash
# Define number of iterations (i)
i=10
#
# Longitudinal drone dynamics benchmark
echo -e "++++++++ RUN DRONE BENCHMARK ++++++++\n";
python3 RunFile.py --model_file AAAI23_models --model 'drone' --noise_samples 20000 --prism_java_memory 8 --input_min -5 --input_max 5 --partition_num_elem 24 20 --iterations $i;
python3 RunFile.py --model_file AAAI23_models --model 'drone' --noise_samples 20000 --prism_java_memory 8 --input_min -5 --input_max 5 --partition_num_elem 24 20 --drone_par_uncertainty --iterations $i;
python3 RunFile.py --model_file AAAI23_models --model 'drone' --noise_samples 20000 --prism_java_memory 8 --input_min -5 --input_max 5 --partition_num_elem 24 20 --drone_spring --iterations $i;
python3 RunFile.py --model_file AAAI23_models --model 'drone' --noise_samples 20000 --prism_java_memory 8 --input_min -5 --input_max 5 --partition_num_elem 24 20 --drone_spring --drone_par_uncertainty --iterations $i;
#
# Building temperature control problem, without epistemic uncertainty
echo -e "++++++++ RUN TEMPERATURE CONTROL BENCHMARK (NO EPISTEMIC UNCERTAINTY) ++++++++\n";
python3 RunFile.py --model_file AAAI23_models --model 'building_temp' --noise_samples 20000 --prism_java_memory 8 --input_min 15 --input_max 30 --partition_num_elem 15 25 --bld_target_size '[[-.2, .2], [-.5, .5]]' --iterations $i;
python3 RunFile.py --model_file AAAI23_models --model 'building_temp' --noise_samples 20000 --prism_java_memory 8 --input_min 15 --input_max 30 --partition_num_elem 25 35 --bld_target_size '[[-.1, .1], [-.3, .3]]' --iterations $i;
python3 RunFile.py --model_file AAAI23_models --model 'building_temp' --noise_samples 20000 --prism_java_memory 8 --input_min 15 --input_max 30 --partition_num_elem 35 45 --bld_target_size '[[-.1, .1], [-.3, .3]]' --iterations $i;
python3 RunFile.py --model_file AAAI23_models --model 'building_temp' --noise_samples 20000 --prism_java_memory 8 --input_min 15 --input_max 30 --partition_num_elem 50 70 --bld_target_size '[[-.05, .05], [-.15, .15]]' --iterations $i;
python3 RunFile.py --model_file AAAI23_models --model 'building_temp' --noise_samples 20000 --prism_java_memory 8 --input_min 15 --input_max 30 --partition_num_elem 70 100 --bld_target_size '[[-.05, .05], [-.15, .15]]' --iterations $i;
#
# Building temperature control problem, with epistemic uncertainty
echo -e "++++++++ RUN TEMPERATURE CONTROL BENCHMARK (WITH EPISTEMIC UNCERTAINTY) ++++++++\n";
python3 RunFile.py --model_file AAAI23_models --model 'building_temp' --noise_samples 20000 --prism_java_memory 8 --input_min 15 --input_max 30 --bld_control_error --partition_num_elem 15 25 --bld_target_size '[[-.2, .2], [-.5, .5]]' --iterations $i;
python3 RunFile.py --model_file AAAI23_models --model 'building_temp' --noise_samples 20000 --prism_java_memory 8 --input_min 15 --input_max 30 --bld_control_error --partition_num_elem 25 35 --bld_target_size '[[-.1, .1], [-.3, .3]]' --iterations $i;
python3 RunFile.py --model_file AAAI23_models --model 'building_temp' --noise_samples 20000 --prism_java_memory 8 --input_min 15 --input_max 30 --bld_control_error --partition_num_elem 35 45 --bld_target_size '[[-.1, .1], [-.3, .3]]' --iterations $i;
python3 RunFile.py --model_file AAAI23_models --model 'building_temp' --noise_samples 20000 --prism_java_memory 8 --input_min 15 --input_max 30 --bld_control_error --partition_num_elem 50 70 --bld_target_size '[[-.05, .05], [-.15, .15]]' --iterations $i;
python3 RunFile.py --model_file AAAI23_models --model 'building_temp' --noise_samples 20000 --prism_java_memory 8 --input_min 15 --input_max 30 --bld_control_error --partition_num_elem 70 100 --bld_target_size '[[-.05, .05], [-.15, .15]]' --iterations $i;
#
# Anaesthesia delivery problem
echo -e "++++++++ RUN ANAESTHESIA DELIVERY BENCHMARK ++++++++\n";
python3 RunFile.py --model_file AAAI23_models --model 'anaesthesia_delivery' --noise_samples 20000 --prism_java_memory 32 --input_min -10 --input_max 40 --partition_num_elem 25 40 40;
