#!/bin/bash
# Define number of iterations (i)
i=2
#
# Longitudinal drone dynamics benchmark
echo -e "++++++++ RUN DRONE BENCHMARK ++++++++\n";
python3 RunFile.py --model 'oscillator' --prism_java_memory 8 --iterations $i;
python3 RunFile.py --model 'oscillator' --prism_java_memory 8 --osc_par_uncertainty --iterations $i;
python3 RunFile.py --model 'oscillator' --prism_java_memory 8 --osc_spring --iterations $i;
python3 RunFile.py --model 'oscillator' --prism_java_memory 8 --osc_spring --osc_par_uncertainty --iterations $i;
#
# Building temperature control problem, without epistemic uncertainty
echo -e "++++++++ RUN TEMPERATURE CONTROL BENCHMARK (NO EPISTEMIC UNCERTAINTY) ++++++++\n";
python3 RunFile.py --model 'building_1room' --prism_java_memory 8 --bld_partition [15,25] --iterations $i;
python3 RunFile.py --model 'building_1room' --prism_java_memory 8 --drug_partition [25,35] --iterations $i;
python3 RunFile.py --model 'building_1room' --prism_java_memory 8 --drug_partition [35,45] --iterations $i;
python3 RunFile.py --model 'building_1room' --prism_java_memory 8 --drug_partition [50,70] --iterations $i;
python3 RunFile.py --model 'building_1room' --prism_java_memory 8 --drug_partition [70,100] --iterations $i;
#
# Building temperature control problem, with epistemic uncertainty
echo -e "++++++++ RUN TEMPERATURE CONTROL BENCHMARK (WITH EPISTEMIC UNCERTAINTY) ++++++++\n";
python3 RunFile.py --model 'building_1room' --prism_java_memory 8 --bld_partition [15,25] --iterations $i --bld_control_error;
python3 RunFile.py --model 'building_1room' --prism_java_memory 8 --drug_partition [25,35] --iterations $i --bld_control_error;
python3 RunFile.py --model 'building_1room' --prism_java_memory 8 --drug_partition [35,45] --iterations $i --bld_control_error;
python3 RunFile.py --model 'building_1room' --prism_java_memory 8 --drug_partition [50,70] --iterations $i --bld_control_error;
python3 RunFile.py --model 'building_1room' --prism_java_memory 8 --drug_partition [70,100] --iterations $i --bld_control_error;
#
# Anaesthesia delivery problem
echo -e "++++++++ RUN ANAESTHESIA DELIVERY BENCHMARK ++++++++\n";
python3 RunFile.py --model 'anaesthesia_delivery' --prism_java_memory 32 --drug_partition [20,30,30];