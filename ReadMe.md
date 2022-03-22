# Introduction of this ReadMe file

This artefact contains the source code for our AAAI 2022 paper:

<center>Thom Badings, Alessandro Abate, David Parker, Nils Jansen, Hasan Poonawala & Marielle Stoelinga (2021). Sampling-based Robust Control of Autonomous Systems with Non-Gaussian Noise. AAAI 2022</center>

This folder contains everything that is needed to replicate the results presented in the paper. Our simulations ran on a Linux machine with 32 3.7GHz cores and 64 GB of RAM. Using the instructions below, the experiments may be replicated on a virtual machine, or on your own machine.

Python version: `3.8.8`. For a list of the required Python packages, please see the `requirements.txt` file. 

------



# Table of contents

[TOC]

------



# Installation and execution of the program

We tested the artefact with the *TACAS 21 Artefact Evaluation VM - Ubuntu 20.04 LTS*, which can be downloaded from the links specified in steps 1 and 2 below. Please follow the instructions below if you want to use the artefact within a virtual machine. 

If you plan to use the program directly on your own machine, you may skip steps 1 and 2.

**<u>Important note:</u>** the PRISM version that we use only runs on MacOS or Linux.

## 1. Download and install VirtualBox

To use the artefact on a virtual machine, first download and install the *VirtualBox* host from the following webpage:

https://www.virtualbox.org/wiki/Downloads

The artefact has been tested with *VirtualBox 6.1.18* on *Windows 10*. 

## 2. Download the TACAS 21 Virtual Machine

We tested the artefact on the *TACAS 21 Artefact Evaluation VM*, which can be downloaded from Zenodo.org:

https://zenodo.org/record/4041464

Download the `TACAS 21.ova` file (size 3.6 GB), then open the VirtualBox application, and import this file by clicking `File` -> `Import Virtual Appliance`. In this menu, select the `TACAS 21.ova` file and click `next`. In the menu that follows, you may change the assigned RAM memory. 

**<u>Note:</u>** in order to run the larger benchmarks, you may want to increase the RAM memory, e.g. to 8192 MB.

After setting the desired settings, click `import` to import the appliance (this may take a few minutes). When this is finished, boot up the virtual machine.

Note that other virtual machines that support Python 3 may work as well, but we tested the artefact specifically for this one.

## 3. Install dependencies

In addition to Python 3 (which is installed on the TACAS 21 virtual machine by default), a number of dependencies must be installed on your (virtual) machine:

1. Git - Can be installed using the command:

   ```bash
   $ sudo apt update 
   $ sudo apt install git
   ```

   **<u>Note:</u>** when asked for a password, the default login and password are `tacas21` / `tacas21`.

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
   
4. To create the 3D UAV trajectory plots, you may need to install a number of libraries requires for Qt, which can be done using the command:

   ```bash
   $ sudo apt-get install -y libdbus-1-3 libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 libxcb-xinput0 libxcb-xfixes0
   ```

## 4. Copy artefact files and install packages

Download and extract the artefact files to a folder on the virtual machine (or on your own machine, if you are not using a virtual one) with writing access (needed to store results).

Open a terminal and navigate to the artefact folder. Then, run the following command to install the required packages:

```bash
$ pip3 install -r requirements.txt
```

Please checkout the file `requirements.txt` to see the full list of packages that will be installed.

## 5. Set default folders and options

To ensure that PRISM can be found by the script, **you need to modify the path to the PRISM folder** in the  `options.txt` file. Set the PRISM folder to the one where you installed it (the filename should end with `/prism/`, such that it points the folder in which the `bin/` folder is located), and save your changes. For example, this line of the `options.txt` file can look like this:

```
/home/<location-to-prism>/prism-imc/prism/
```

If desired, you may also make other changes in the configuration of the script in the `options.txt` file. An overview of the most important settings is given below:

- `scenarios.samples` : the number of samples the script uses in the first iteration
- `scenarios.samples_max` : the number of samples after which the iterative scheme is terminated
- `scenarios.confidence` : the confidence level used for computing transition probability intervals
- `mdp.prism_folder` : folder where PRISM is located; should end with `/prism/` (the folder in which the `bin/` folder is located)
- `mdp.mode` : if “*interval*”, an interval MDP is created. If “*estimate*”, a regular MDP is created
- `mdp .prism_model_writer` : if “*explicit*”, a PRISM model is created in explicit form. If “*default*”, a standard PRISM model is created. See the PRISM documentation for more details.
- `mdp.prism_java_memory` : the memory allocated to Java when running PRISM. The default value is 2 GB, but when solving large models, this may be increased (the benchmarks in the paper all ran on a machine with 32 GB of memory allocated to Java).
- `main.iterative` : if True, the iterative scheme is enabled; if False, it is disabled
- `plotting.partitionPlot` : if True, a 2D plot of the partition is created; if False, this plot is not created
- `plotting.3D_UAV` : if True, the 3D plots for the 3D UAV benchmark are created. Note that **<u>this plot pauses the script until it is closed</u>**. If you do not want this behaviour, you need to disable this option.
- `scenarios.gamma` : the factor by which the number of samples is multiplied after each iteration of the iterative abstraction scheme.

## 6. Run the script

Run the `SBA-RunFile.py` file to execute the program, by typing the command:

```bash
$ python SBA-RunFile.py
```

You will be asked to make a number of choices:

1. The **application** (i.e. benchmark) you want to work with (see below for details on how to add a model). For some applications, you will be asked an additional question, such as the dimension (for the UAV case) and grid size (for the 1-zone BAS case).
   - **<u>Note 1:</u>** the 2-zone BAS application is quite memory intense, and could take a few hours to run for all iterations (depending on your machine). If you want to run a smaller case, please consider choosing the 2D UAV or 1-zone BAS application.
   - **<u>Note 2:</u>** If you experience problems with creating the **3D trajectory plots** for the 3D UAV application, you can disable it by setting `plotting.3D_UAV = False` in the `options.txt` file (see Section 5).
2. Whether you want to run **Monte Carlo** simulations. If chosen to do so, you are asked an additional question to fill in the **number of Monte Carlo simulations** to run.
3. Whether you want to **start a new abstraction** or **load existing PRISM results**.

The user can recreate the results presented in the paper, by choosing the **3D UAV** application or the **2-zone building automation system (BAS)** application.

**<u>Important note:</u>** after every iteration of the 3D UAV case, an interactive 3D plot is created with `visvis`, that shows a number of trajectories under the optimal controller. **<u>This plot will pause the script</u>**, until it is closed manually by the user. Note that a screenshot of the plot is automatically stored in the results folder of the current iteration.

## 7. Inspect the results 

All results are stored in the `output/` folder. When running `SBA-RunFile.py` for a new abstraction, a new folder is created that contains the application name and the current datetime, such as `ScAb_UAV_06-02-2021_13-46-29/`.

Results that apply to all iterations of the iterative abstraction scheme are saved directly in this folder. This includes an Excel file describing all model sizes, run times, etc.

For every iteration, a subfolder is created based on the number of samples that was used, e.g. `N=3200`. Within this subfolder, all results specific to that single iteration are saved. This includes:

- The PRISM model files. Depending on the mode, this can either be in explicit format (in which case a `.lab`, `.sta`, and `.tra` file are created), or as a single file if the default mode is selected.
- An Excel file that describes all results, such as the optimal policy, model size, run times, etc., of the current iteration.
- Various plots, showing the appropriate results for the current iteration.

------



# Adding or modifying model definitions

You can add models or change the existing ones by modifying the file `core/modelDefinitions.py`. Every application is defined as a class, having the application name as the class name. An application class has two functions:

### Initializing function

Here, the basic setup of the application is defined. This includes the following elements:

- `setup['deltas']` (list of integers) is assigned a list of integers (with only one value by default). It denotes the number of discrete time steps that are grouped together, to render the dynamical system fully actuated. For example, if the dimension of the state is 6, and the dimension of the control input is 3, then it will be `setup['deltas'] = [2]`.
- `setup['control']['limits'] ['uMin'] / ['uMax']` (list of integers) are the control authority limits. It is given as a list, and every entry reflects a dimension of the state.
- `setup['partition']['nrPerDim']` (list of integers) is the number of regions defined in every dimension of the state space. Note that the partition has a region centred at the origin when odd values are given.
- `setup['partition']['width']` (list of floats) is the width of the regions in every dimension of the state space.
- `setup['partition']['origin']` (list of floats) denotes the origin of the partitioned region of the state space. 
- `setup['targets']['nrPerDim']` (either `'auto'` or a list of integers) is the number of target points defined in every dimension of the state. When `'auto'` is given, the number of target points equals the number of regions of the partition.
- `setup['targets']['domain']` (either `'auto'` or a list of integers) is the domain over which target points are defined. When `'auto'` is given, the domain is set equal to that of the state space partition.
- `setup['specification']['goal']` (nested list of floats) is the list of points whose associated partitioned regions are in the goal region. The function `setStateBlock()` is a helper function to easily define blocks of goal regions, by creating slices in the state space partition in given dimensions.
- `setup['specification']['critical']` (nested list of floats) is the list of points whose associated partitioned regions are in the critical region. The function `setStateBlock()` is a helper function to easily define blocks of critical regions, by creating slices in the state space partition in given dimensions.
- `tau` (float) is the time discretization step size.
- `setup['endTime']` (integer > 0) is the finite time horizon over which the reach-avoid problem is solved.

### SetModel function

In the `setModel` function, the linear dynamical system is defined in the following form (see the submitted paper for details): 
$\mathbf{x}_{k+1} = A \mathbf{x}_k + B \mathbf{u}_k + \mathbf{q}_k + \mathbf{w}_k,$
where:

- `A` is an n x n matrix.
- `B` is an n x p matrix.
- `Q` is a n x 1 column vector that reflects the additive deterministic disturbance (q-term in the equation above).
- If Gaussian noise is used, `noise['w_cov']` is the covariance matrix of the w-term in the equation above. Note that non-Gaussian noise (from the Dryden gust model) is used for the UAV case.

Note that is the current version of the codes is not compatible (yet) with partial observability (i.e. defining an observer). Thus, make sure to set the argument `observer = False`.

For some models, the model definition is given in non-discretized form, i.e.
$$
\dot{\mathbf{x}}(t) = A_c\mathbf{x}(t) + B_c\mathbf{u}(t) + \mathbf{q}_c(t) + \mathbf{w}_c(t),
$$
where subscript c indicates that these matrices and vectors differ from the ones above. If a continuous-time dynamical model is given, it is discretized using one of two methods:

- Using a forward Euler method.
- Using a Gears discretization method.

------



# Ancillary scripts

In addition to the main Python program which is executed using `SBA-RunFile.py`, there are two ancillary scripts contained in the folder:

### MatLab code to tabulate probability intervals

We provide a convenient MatLab script, called `Tabulate-RunFile.m`, which can be used to tabulate all possible transition probability intervals for a given value of `N` (total number of samples) and `beta` (the confidence level). For more details on how the transition probability intervals are computed, please consult the main paper (and in particular Theorem 1).

For every combination of `N` and `beta`, the script creates a `.csv` file, that contains the tabulated transition probability intervals, e.g., named `probabilityTable_N=3200_beta=0.01.csv`. When running the main Python program for these values of `N` and `beta`, the tabulated data is loaded into Python, to compute the transition probability intervals of the interval MDP.

### Python code to create turbulence samples

The Python script `createTurbulenceSamples` can be used to create (non-Gaussian) noise samples for the 3D UAV case. The source code for the Dryden gust model used to create these samples, can be found in `core/UAV/dryden.py`. The script stored the samples in the `input/` folder, in a `.csv` file that contains the number of samples is contain in its name, e.g., `TurbulenceNoise_N=100.csv`.
