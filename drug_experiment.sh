#!/bin/bash
python3 RunFile.py --model 'anaesthesia_delivery' --prism_java_memory 32 --drug_partition [5,5,5];
python3 RunFile.py --model 'anaesthesia_delivery' --prism_java_memory 32 --drug_partition [20,30,30];
python3 RunFile.py --model 'anaesthesia_delivery' --prism_java_memory 32 --drug_partition [25,40,40];
python3 RunFile.py --model 'anaesthesia_delivery' --prism_java_memory 32 --drug_partition [25,50,50];
python3 RunFile.py --model 'anaesthesia_delivery' --prism_java_memory 32 --drug_partition [30,60,60];
python3 RunFile.py --model 'anaesthesia_delivery' --prism_java_memory 32 --drug_partition [40,75,75];