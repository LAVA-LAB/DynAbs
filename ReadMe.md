# Introduction of this ReadMe file

DynAbs is a tool that can be used to compute feedback controllers for stochastic linear dynamical systems with reach-avoid control tasks.
The returned controllers are provably correct, which means that they satisfy the reach-avoid task *with at least a desired threshold probability.*

More specifically, this artefact contains an implementation of the methods proposed in the following papers:

- [Thom Badings, Alessandro Abate, David Parker, Nils Jansen, Hasan Poonawala & Marielle Stoelinga (2022). Sampling-based Robust Control of Autonomous Systems with Non-Gaussian Noise. AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/21201)
- [Thom Badings, Licio Romao, Alessandro Abate, David Parker, Hasan Poonawala, Marielle Stoelinga & Nils Jansen (2022). Robust Control for Dynamical Systems with Non-Gaussian Noise via Formal Abstractions. JAIR 2023](https://www.jair.org/index.php/jair/article/view/14253)
- [Thom Badings, Licio Romao, Alessandro Abate, & Nils Jansen (2023). Probabilities Are Not Enough: Formal Controller Synthesis for Stochastic Dynamical Models with Epistemic Uncertainty. AAAI 2023](https://arxiv.org/pdf/2210.05989.pdf)

This repository contains all code and instructions that are needed to replicate the results presented in the paper. Our simulations ran on a Linux machine with 32 3.7GHz cores and 64 GB of RAM.

Python version: `3.10.14`. For a list of the required Python packages, please see the `requirements.txt` file. The code is tested with PRISM version `4.8.1`.

------


# Installation and execution of the program

> **<u>Important note:</u>** We have only tested the artefact on MacOS and Linux. Windows is currently not supported (as we require building PRISM from source).

We recommend using the artefact on a virtual environment, in order to keep things clean on your machine. Here, we explain how to install the artefact on such a virtual environment using Conda. Other methods for using virtual environments exist, but we assume that you have Python 3 installed (we tested with version `3.10.14`).

## 1. Create virtual environment

To create a virtual environment with Conda and the prefered Python version, run the following command (replacing `abstract_env` with the desired name for the environment):

```bash
conda create --name abstract_env python=3.10.14
```

Then, to activate the virtual environment, run:

```bash
conda activate abstract_env
```

## 2. Install dependencies

We assume the git is installed on your machine. Then, the following dependencies must be installed on your machine:

1. Java Development Kit (required to run PRISM, see https://prismmodelchecker.org/manual/InstallingPRISM/Instructions for details). On Linux, this can be installed using the commands:

   ```bash
   sudo apt install default-jdk
   ```

   On MacOS, the Java Development kit can be installed via, for example, Homebrew or Oracle (https://www.oracle.com/java/technologies/downloads/#jdk22-mac).

3. PRISM - In the desired PRISM installation folder, clone PRISM (version `4.8.1`) from git and run the makefile:

   ```bash
   git clone https://github.com/prismmodelchecker/prism.git prism --branch v4.8.1;
   cd prism/prism; 
   make
   ```

   For more details on installing and using PRISM, we refer to the PRISM documentation on 
   https://www.prismmodelchecker.org
   
5. To create the 3D UAV trajectory plots, you may need to install a number of libraries required for Qt. On Linux, these libraries can be installed using the command:

   ```bash
   sudo apt-get install -y libdbus-1-3 libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 libxcb-xinput0 libxcb-xfixes0
   ```

   On MacOS, the Python requirements should be sufficient to generate the 3D plots. If you run into any problems, feel free to contact us (see contact details at the bottom of the ReadMe).

## 3. Copy artefact files and install packages

Download and extract the artefact files to a folder on your machine, making sure that you grant writing access (needed to store results).

Open a terminal and navigate to the artefact folder. Then, run the following command to install the required packages:

```bash
pip3 install -r requirements.txt
```

Please checkout the file `requirements.txt` to see the full list of packages that will be installed.

------

# How to run for a single model?

An example of running the UAV benchmark (6D linear dynamical model) is as follows:

```bash
python3 RunFile.py --model_file JAIR22_models --model UAV --prism_executable '~/Documents/prism/bin/prism' --UAV_dim 3 --prism_java_memory 8 --noise_samples 6400 --noise_factor 1 --nongaussian_noise --monte_carlo_iter 1000 --x_init '[-14,0,6,0,-2,0]' --plot --timebound 16
```

This runs the 3D UAV benchmark from the paper, with `N=6400` (non-Gaussian) noise samples, and Monte Carlo simulations enabled.

> Importantly, you have to change the `prism_executable` argument such that it points to the Prism executable for you. This is the file named `prism` in the `.../prism/bin/` folder.

All results are stored in the `output/` folder. When running `RunFile.py` for a new abstraction, a new folder is created that contains the application name and the current datetime, such as `Ab_UAV_09-21-2022_17-31-20/`. For every iteration, a subfolder is created, inwhich all results specific to that single iteration are saved. This includes:

- The PRISM model files (namely a `.lab`, `.sta`, and `.tra` file).
- An Excel file that describes all results, such as the optimal policy, model size, run times, etc., of the current iteration.
- Various plots, showing the appropriate results for the current iteration.

------

# How to run experiments from the paper?

The figures and tables in the experimental section of the paper can be reproduced by running the shell script `run_experiments.sh` in the root folder of the repository.
To run the experiments, change the `prism_executable` variable such that it points to the Prism executable on your machine (also see the information above), and then run the following to performe the experiments presented in the respective papers:

```bash
bash run_experiments_JAIR23.sh;
bash run_experiments_AAAI23.sh
```

------

# What arguments can be passed?

Below, we list all arguments that can be passed to the command for running the program. Arguments are given as `--<argument name> <value>`. Note that only the `model` argument is required; all others are optional (and have certain default values).

| Argument           | Required? | Default            | Type                     | Description |
| ---                | ---       | ---                | ---                      | ---         |
| model_file         | Yes       | N/A                | str                      | File from which to load model, without `.py` (by default, `AAAI23_models` and `JAIR22_models` are supplied) |
| model              | Yes       | N/A                | str                      | Name of the model to load |
| timebound          | Yes       | inf                | int or 'inf'             | Timebound on the specification/property, which can be a positive integer or 'inf' for an infinite horizon (unbounded) |
| mdp_mode           | No        | interval            | str                      | If `estimate`, a point estimate MDP abstraction is created; if `interval`, a robust interval MDP abstraction is created |
| abstraction_type   | No        | default            | str                      | If `default`, no set-bounded parameter uncertainty is considered; if `parameter`, set-bounded parameter uncertainty is considered next to stochastic noise |
| noise_samples      | No        | 20000              | int                      | Number of noise samples to use for computing transition probability intervals |
| confidence         | No        | 1e-8               | float                    | Confidence level on individual transitions |
| sample_clustering  | No        | 1e-2               | float                    | Distance at which to cluster (merge) similar noise samples |
| prism_java_memory  | No        | 1                  | int                      | Max. memory usage by JAVA / PRISM |
| iterations         | No        | 1                  | int                      | Number of repetitions of computing iMDP probability intervals |
| nongaussian_noise  | No        | False              | Boolean (no value)       | If argument `--nongaussian_noise` is passed, use non-Gaussian noise samples (used for UAV benchmark) |
| monte_carlo_iter   | No        | 0                  | int                      | Number of Monte Carlo simulations to perform |
| plot               | No        | False              | Boolean (no value)       | If argument `--plot` is passed, plots are created in general |
| partition_plot     | No        | False              | Boolean (no value)       | If argument `--partition_plot` is passed, create partition plot |
| x_init             | No        | []                 | List                     | Initial state for Monte Carlo simulations |
| verbose            | No        | False              | Boolean (no value)       | If argument `--verbose` is passed, more verbose output is provided by the script |

------

# Contact

In case you have questions or experience problems with the code, feel free to reach out to Thom Badings (thombadings@gmail.com).