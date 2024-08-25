# Introduction of this ReadMe file

This branch contains an implementation of the formal abstraction method proposed in the ECC 2024 paper:

- [Thom Badings, Licio Romao, Alessandro Abate, & Nils Jansen (2024). A Stability-Based Abstraction Framework for Reach-Avoid Control of Stochastic Dynamical Systems with Unknown Noise Distributions. ECC 2024](https://arxiv.org/pdf/2404.01726)

> **Note:** Please see the main branch for the code corresponding to other papers.

This repository contains all code and instructions that are needed to replicate the results presented in the paper. Our simulations ran on a Linux machine with 32 3.7GHz cores and 64 GB of RAM.

Python version: `3.8.8`. For a list of the required Python packages, please see the `requirements.txt` file. 

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

4. To create the 3D UAV trajectory plots, you may need to install a number of libraries required for Qt, which can be done using the command:

   ```bash
   $ sudo apt-get install -y libdbus-1-3 libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0                          libxcb-render-util0 libxcb-xinerama0 libxcb-xinput0 libxcb-xfixes0
   ```

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

An example of running the UAV benchmark (6D linear dynamical model) is as follows:

```bash
$ python3 RunFile.py --model_file JAIR22_models --model UAV --UAV_dim 3 --prism_java_memory 8 --noise_samples 6400 --noise_factor 1 --nongaussian_noise --monte_carlo_iter 1000 --x_init '[-14,0,6,0,-2,0]' --plot --timebound 16
```

This runs the 3D UAV benchmark from the paper, with `N=6400` (non-Gaussian) noise samples, and Monte Carlo simulations enabled.

All results are stored in the `output/` folder. When running `RunFile.py` for a new abstraction, a new folder is created that contains the application name and the current datetime, such as `Ab_UAV_09-21-2022_17-31-20/`. For every iteration, a subfolder is created, inwhich all results specific to that single iteration are saved. This includes:

- The PRISM model files (namely a `.lab`, `.sta`, and `.tra` file).
- An Excel file that describes all results, such as the optimal policy, model size, run times, etc., of the current iteration.
- Various plots, showing the appropriate results for the current iteration.

------

# How to run experiments from the ECC 2024 paper?

The figures and tables in the experimental section of the paper can be reproduced by running the shell script `run_stability_experiments.sh` in the root folder of the repository:

```bash
bash run_stability_experiments.sh
```
