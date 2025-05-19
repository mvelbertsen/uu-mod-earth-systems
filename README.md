# Modelling Earth Systems - Project Code
A python reworking of the visco-elasto-plastic geodynamics code developed in the book *Numerical Geodynamic Modelling* by Taras Gerya [1]. It uses a finite difference staggered grid approach to solve the Stoke's, continuity and temperature equations, along with markers for advection.

## Installation
The required python packages are listed in the `environment.yml` file.  To create a new conda environment with these packages installed, run the command `conda env create -f environment.yml` in the top directory of the repo.  This will create an environment called `mod-earth`, which can then be activated using `conda activate mod-earth`.  

## Running a model
The file `main.py` contains the central code, to run a model you can simply run this file. 

Various model setups can be found in the `models` directory.  In each sub-directory you will find two files: `setup.py` and `material_properties*.txt`.  These contain the code and required material parameters for that model, in a function called `initializeModel`, which is called by `main.py` to setup the simulation.  

To run a given setup, change the import statement for the `initializeModel` function in `main.py` to point to the directory of the model you wish to run.  For example, to run the model in the directory `lithosphereExtension`, we change the import statement in `main.py` to `from models.lithosphereExtension.setup import initializeModel`.

## Creating your own model
To create your own setup, you must create a new directory in `models` and populate it with an implementation of `setup.py` and `material_properties*.txt`.  `setup.py` must contain the function `initializeModel` (see the docstring of the example implementations for details on what it should return), which sets the initial conditions, boundary conditions and parameter values for the simulation.  We recommend that you start by copying an existing `setup.py` and then modifying it for your own implementation.

## Authors
__Charlotte Summers__  
Research Support Officer, Department of Earth Sciences
Utrecht University 
email: c.summers [at] uu.nl  

## References
[1] Gerya T. MATLAB program examples. In: Introduction to Numerical Geodynamic Modelling. Cambridge University Press; 2019:425-437. DOI: https://doi.org/10.1017/9781316534243.024 
