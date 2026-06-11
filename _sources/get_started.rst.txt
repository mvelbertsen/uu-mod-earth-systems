Getting Started
===============

Prerequistes
------------

This code requires Python 3.14 or higher (earlier versions may work but have not been tested), so if you haven't already installed Python, do so before continuing.

The following guide assumes you are using the Anaconda distribution of Python (the one we use in this course).

Installation
------------

Download the code from GitHub using either ``git clone URL-of-the-repository`` or just by clicking the green 'Code' button in the top-right of the page and then 'Download ZIP'.

The required packages for this code are listed in the ``environment.yml`` file, which can be used to create a Python virtual environment and install these dependencies in it.

Using the command line
^^^^^^^^^^^^^^^^^^^^^^

Open a terminal (on MacOS or Linux) or on Windows the ``anaconda_prompt`` app from the Anaconda Navigator. 
Navigate to the directory with the code in using 

.. code-block:: bash

    cd /path/to/directory/on/your/computer/uu-mod-earth-systems

(if using Anaconda) Run the command

.. code-block:: bash

    conda env create -f environment.yml

This will create a conda environment called ``modearth-project``, which you can then activate using the command

.. code-block:: bash

    conda activate modearth-project

The terminal prompt should now display ``(modearth-project)`` in front of the entry field.  You can then launch the ``spyder`` IDE in the correct environment by running

.. code-block:: bash

    spyder

in this terminal.  You can then develop and run models within this spyder instance.

Using Anaconda Navigator
^^^^^^^^^^^^^^^^^^^^^^^^

In the environments tab, click on "Import" then select the ``environment.yml`` file and select "Create".

Running an Example Model
------------------------

To check that you can run the code, try running one of the example models, which are found in the ``models`` directory.
To run a model with it's default settings, simply run the file ``run.py`` from it's directory.

SimpleStokes
^^^^^^^^^^^^

The ``SimpleStokes`` model is a simple test case, where the domain is split into two vertically and the 
material on each side of the domain has a different density.  To run the model, run the ``run.py`` file in the ``SimpleStokes`` directory.

The code includes plotting, for the default parameters you should see that a new folder ``Results/figures/SimpleStokes`` has
been created in the top level directory of the repository.  You should see plot every two timesteps in there, showing that a 
rotational flow is set up and the two different materials begin to overturn, as shown here

.. image:: /images/SimpleStokes_litho_18.png 
