#!/bin/bash
# Define number of iterations (i)
i=10
prism_exe='/Users/tbadings/Documents/Code_temp/prism/prism/bin/prism'
#
# Longitudinal drone dynamics benchmark
echo -e "++++++++ RUN DRONE BENCHMARK ++++++++\n";
python3 RunFile.py --abstraction_type 'parameter' --model_file AAAI23_models --model 'drone' --prism_executable $prism_exe --noise_samples 20000 --prism_java_memory 8 --iterations $i;
python3 RunFile.py --abstraction_type 'parameter' --model_file AAAI23_models --model 'drone' --prism_executable $prism_exe --noise_samples 20000 --prism_java_memory 8 --drone_par_uncertainty --iterations $i;
python3 RunFile.py --abstraction_type 'parameter' --model_file AAAI23_models --model 'drone' --prism_executable $prism_exe --noise_samples 20000 --prism_java_memory 8 --drone_spring --iterations $i;
python3 RunFile.py --abstraction_type 'parameter' --model_file AAAI23_models --model 'drone' --prism_executable $prism_exe --noise_samples 20000 --prism_java_memory 8 --drone_spring --drone_par_uncertainty --iterations $i;
#
# Building temperature control problem, without set-bounded parameter uncertainty
echo -e "++++++++ RUN TEMPERATURE CONTROL BENCHMARK (NO EPISTEMIC UNCERTAINTY) ++++++++\n";
python3 RunFile.py --abstraction_type 'parameter' --model_file AAAI23_models --model 'building_temp' --prism_executable $prism_exe --noise_samples 20000 --prism_java_memory 8 --bld_partition '[15,25]' --bld_target_size '[[-.2, .2], [-.5, .5]]' --iterations $i;
python3 RunFile.py --abstraction_type 'parameter' --model_file AAAI23_models --model 'building_temp' --prism_executable $prism_exe --noise_samples 20000 --prism_java_memory 8 --bld_partition '[25,35]' --bld_target_size '[[-.1, .1], [-.3, .3]]' --iterations $i;
python3 RunFile.py --abstraction_type 'parameter' --model_file AAAI23_models --model 'building_temp' --prism_executable $prism_exe --noise_samples 20000 --prism_java_memory 8 --bld_partition '[35,45]' --bld_target_size '[[-.1, .1], [-.3, .3]]' --iterations $i;
python3 RunFile.py --abstraction_type 'parameter' --model_file AAAI23_models --model 'building_temp' --prism_executable $prism_exe --noise_samples 20000 --prism_java_memory 8 --bld_partition '[50,70]' --bld_target_size '[[-.05, .05], [-.15, .15]]' --iterations $i;
python3 RunFile.py --abstraction_type 'parameter' --model_file AAAI23_models --model 'building_temp' --prism_executable $prism_exe --noise_samples 20000 --prism_java_memory 32 --bld_partition '[70,100]' --bld_target_size '[[-.05, .05], [-.15, .15]]' --iterations $i;
#
# Building temperature control problem, with set-bounded parameter uncertainty
echo -e "++++++++ RUN TEMPERATURE CONTROL BENCHMARK (WITH EPISTEMIC UNCERTAINTY) ++++++++\n";
python3 RunFile.py --abstraction_type 'parameter' --model_file AAAI23_models --model 'building_temp' --prism_executable $prism_exe --noise_samples 20000 --prism_java_memory 8 --bld_par_uncertainty --bld_partition '[15,25]' --bld_target_size '[[-.2, .2], [-.5, .5]]' --iterations $i;
python3 RunFile.py --abstraction_type 'parameter' --model_file AAAI23_models --model 'building_temp' --prism_executable $prism_exe --noise_samples 20000 --prism_java_memory 8 --bld_par_uncertainty --bld_partition '[25,35]' --bld_target_size '[[-.1, .1], [-.3, .3]]' --iterations $i;
python3 RunFile.py --abstraction_type 'parameter' --model_file AAAI23_models --model 'building_temp' --prism_executable $prism_exe --noise_samples 20000 --prism_java_memory 8 --bld_par_uncertainty --bld_partition '[35,45]' --bld_target_size '[[-.1, .1], [-.3, .3]]' --iterations $i;
python3 RunFile.py --abstraction_type 'parameter' --model_file AAAI23_models --model 'building_temp' --prism_executable $prism_exe --noise_samples 20000 --prism_java_memory 8 --bld_par_uncertainty --bld_partition '[50,70]' --bld_target_size '[[-.05, .05], [-.15, .15]]' --iterations $i;
python3 RunFile.py --abstraction_type 'parameter' --model_file AAAI23_models --model 'building_temp' --prism_executable $prism_exe --noise_samples 20000 --prism_java_memory 32 --bld_par_uncertainty --bld_partition '[70,100]' --bld_target_size '[[-.05, .05], [-.15, .15]]' --iterations $i;
#
# Anaesthesia delivery problem
echo -e "++++++++ RUN ANAESTHESIA DELIVERY BENCHMARK ++++++++\n";
python3 RunFile.py --abstraction_type 'parameter' --model_file AAAI23_models --model 'anaesthesia_delivery' --prism_executable $prism_exe --noise_samples 20000 --prism_java_memory 32 --drug_partition [25,40,40];
