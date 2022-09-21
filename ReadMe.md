# Introduction of this ReadMe file

This artefact contains the source code for the AAAI 2023 submission with the title:

- "Probabilities Are Not Enough: Formal Controller Synthesis for Stochastic Dynamical Models with Epistemic Uncertainty"

This repository contains all code and instructions that are needed to replicate the results presented in the paper. Our simulations ran on a Linux machine with 32 3.7GHz cores and 64 GB of RAM.

Python version: `3.8.8`. For a list of the required Python packages, please see the `requirements.txt` file. 

------



# Table of contents

[TOC]

------



# Installation and execution of the program

**<u>Important note:</u>** the PRISM version that we use only runs on MacOS or Linux.

We recommend using the artefact on a virtual environment, in order to keep things clean on your machine. Here, we explain how to install the artefact on such a virtual environment using Conda. Other methods for using virtual environments exist, but we assume that you have Python 3 installed (we tested with version 3.8.8).

## 1. Create virtual environment

To create a virtual environment with Conda, run the following command:

```bash
$ conda create --name abstract_env
```

Then, to activate the virtual environment, run:

```bash
$ conda activate abstract_env
```

## 2. Install dependencies

In addition to Python 3, a number of dependencies must be installed on your machine:

1. Git - Can be installed using the command:

   ```bash
   $ sudo apt update 
   $ sudo apt install git
   ```

2. Java Development Kit (required to run PRISM) - Can be installed using the commands:

   ```bash
   $ sudo apt install default-jdk
   ```

3. PRISM (iMDP branch) - In the desired PRISM installation folder, run the following commands:

   ```bash
   $ git clone -b imc https://github.com/davexparker/prism prism-imc
   $ cd prism-imc/prism; make
   ```

   For more details on using PRISM, we refer to the PRISM documentation on 
   https://www.prismmodelchecker.org

## 3. Copy artefact files and install packages

Download and extract the artefact files to a folder on the machine with writing access (needed to store results).

Open a terminal and navigate to the artefact folder. Then, run the following command to install the required packages:

```bash
$ pip3 install -r requirements.txt
```

Please checkout the file `requirements.txt` to see the full list of packages that will be installed.

## 4. Set path to PRISM

To ensure that PRISM can be found by the script, **you need to modify the path to the PRISM folder** in the  `path_to_prism.txt` file. Set the PRISM folder to the one where you installed it (the filename should end with `/prism/`, such that it points the folder in which the `bin/` folder is located), and save your changes. For example, the path to PRISM can look as follows:

```
/home/<location-to-prism>/prism-imc/prism/
```

------

# How to run for a single model?

A minimal example of running the program is as follows:

```bash
$ python3 RunFile.py --model 'drone'
```

This runs the longitudinal drone dynamics benchmark (see the paper for details), with the default options.

All results are stored in the `output/` folder. When running `SBA-RunFile.py` for a new abstraction, a new folder is created that contains the application name and the current datetime, such as `Ab_drone_08-19-2022_07-46-44/`. For every iteration, a subfolder is created, inwhich all results specific to that single iteration are saved. This includes:

- The PRISM model files (namely a `.lab`, `.sta`, and `.tra` file).
- An Excel file that describes all results, such as the optimal policy, model size, run times, etc., of the current iteration.
- Various plots, showing the appropriate results for the current iteration.

------

# How to run experiments from the paper?

The figures and tables in the experimental section of the paper can be reproduced by running the shell script `run_experiments.sh` in the root folder of the repository:

```bash
bash run_experiments.sh
```

This shell script contains one variable that controls the number of iterations to perform for some experiments. To reduce the computation time, you may lower the number of iterations.

------

# What arguments can be passed?

Below, we list all arguments that can be passed to the command for running the program. Arguments are given as `--<argument name> <value>`. Note that only the `model` argument is required; all others are optional (and have certain default values).

| Argument           | Required? | Default            | Type                     | Description |
| ---                | ---       | ---                | ---                      | ---         |
| model              | Yes       | N/A                | str                      | Name of the model to load (options are: `drone`, `building_temp`, or `anaesthesia_delivery`) |
| noise_samples      | No        | 20000              | int                      | Number of noise samples to use for computing transition probability intervals |
| confidence         | No        | 1e-8               | float                    | Confidence level on individual transitions |
| sample_clustering  | No        | 1e-2               | float                    | Distance at which to cluster (merge) similar noise samples |
| prism_java_memory  | No        | 1                  | int                      | Max. memory usage by JAVA / PRISM |
| iterations         | No        | 1                  | int                      | Number of repetitions of computing iMDP probability intervals |
| monte_carlo_iter   | No        | 0                  | int                      | Number of Monte Carlo simulations to perform |
| partition_plot     | No        | False              | Boolean (no value!)      | If argument `--partition_plot` is passed, create partition plot |
| verbose            | No        | False              | Boolean (no value!)      | If argument `--verbose` is passed, more verbose output is provided by the script |

Moreover, the following arguments can specifically be passed for running one of the benchmarks from the paper.
| Benchmark            | Argument              | Required? | Default            | Type                      | Description |
| ---                  | ---                   | ---       | ---                | ---                       | ---         |
| drone                | drone_spring          | No        | False              | Boolean (no value)        | If `--drone_spring` is passed, the spring coefficient is modelled |
| drone                | drone_par_uncertainty | No        | False              | Boolean (no value)        | If `--drone_par_uncertainty` is passed, enable parameter uncertainty |
| drone                | drone_mc_step         | No        | 0.2                | float                     | Step size in which to increment parameter deviation (to create Figs. 5 and 9 as in paper) |
| drone                | drone_mc_iter         | No        | 0.2                | int                       | Number of Monte Carlo simulations (to create Figs. 5 and 9 as in paper) |
| building_temp        | bld_partition         | No        | '[25,35]'          | str (interpreted as list) | Size of the state space partition |
| building_temp        | bld_target_size       | No        | '[[-0.1,0.1],[-0.3,0.3]]' | str (interpreted as list) | Size of the target sets used |
| building_temp        | bld_par_uncertainty   | No        | False              | Boolean (no value)        | If `--bld_par_uncertainty` is passed, enable parameter uncertainty |
| anaesthesia_delivery | drug_partition        | No        | '[20,20,20]'       | str (interpreted as list) | Size of the state space partition |

------

# Ancillary scripts

In addition to the main Python program which is executed using `SBA-RunFile.py`, there are two ancillary scripts contained in the folder:

### MatLab code to tabulate probability intervals

We provide a convenient MatLab script, called `Tabulate-RunFile.m`, which can be used to tabulate all possible transition probability intervals for a given value of `N` (total number of samples) and `beta` (the confidence level). For more details on how the transition probability intervals are computed, please consult the main paper (and in particular Theorem 1).

For every combination of `N` and `beta`, the script creates a `.csv` file, that contains the tabulated transition probability intervals, e.g., named `probabilityTable_N=3200_beta=0.01.csv`. When running the main Python program for these values of `N` and `beta`, the tabulated data is loaded into Python, to compute the transition probability intervals of the interval MDP.
