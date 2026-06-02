# Modelling Earth Systems - Project Code

A python reworking of the visco-elasto-plastic geodynamics code developed in the book *Numerical Geodynamic Modelling* by Taras Gerya [1]. It uses a finite difference staggered grid approach to solve the Stoke's, continuity and temperature equations, along with markers for advection.

Here we give basic install instructions, but for more details on working with the code take a look at the [documentation](https://csummers25.github.io/uu-mod-earth-systems/index.html).

## Installation

If you haven't already, made sure you have python 3 installed (you can do this by installing Anaconda [here](https://www.anaconda.com/download)).  The required python packages are listed in the `environment.yml` file, you will need to create a new environment using this file.  This can be done either within the Anaconda Navigator or through the command line.  

### Using Anaconda Navigator

Click on the Environments tab, then at the bottom of the list click 'Import'.  Select 'Local Drive' then open the `environment.yml` file included with the code.  Set the environment name to 'modearth-project' and click 'Import'.  The enviroment will likely take a few minutes to be created, once it's finished you can select it from the list of environments to activate it.  

### Using the terminal

Open a terminal (or on Windows the `anaconda_prompt` app from within the Anaconda naviagator), navigate to the directory in which you have downloaded this repository using `cd ./path/to/directory/on/your/computer/uu-mod-earth-systems` and then run the command `conda env create -f environment.yml` in the top directory of the repo.  This will create an environment called `modearth-project`, which can then be activated using `conda activate modearth-project` in the terminal (or by selecting it in the environments tab in Anaconda Navigator).  To launch Spyder in this environment, either launch it from the terminal with the environment activated with the command `spyder`, or activate the environment in Anaconda navigator and select spyder from the list of installed apps.

## Running different model setups

To run a model you can simply run the `run.py` file within that model's directory from your choice of python IDE/interpreter.  The environment installed in the previous step includes the Spyder IDE so we recommend using this.  After activating your environment as described in the install instructions you can launch spyder by typing `spyder` in the terminal (or by lauching it through the Anaconda Navigator once you've activated the correct enviroment in the 'Environments' tab).  You can then open the `run.py` file in spyder and run it using the run icon in the top bar. 

Various model setups can be found in the `models` directory.  In each sub-directory you will find four files: `run.py`, `setup.py`, `material_properties.txt` and `visualisation.py`.  These contain the code and required material parameters for that model.  

## Changing model parameters

In the exercises you will be asked to change parameters for the different model setups.  This is done by editing the `setup.py` and `material_properties.txt` files within that model's directory. To keep track of all your different cases, remember to also change the `parameters.output_name` parameter to something that clearly describes the case you are running.  This will then output your results to a directory with that name within `Results/figures`.  To make different plots, you can edit the `visualisation.py` in the model's directory.

## References
[1] Gerya T. MATLAB program examples. In: Introduction to Numerical Geodynamic Modelling. Cambridge University Press; 2019:425-437. DOI: https://doi.org/10.1017/9781316534243.024 
