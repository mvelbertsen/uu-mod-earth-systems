Running existing models
=======================

Each example model is stored in it's own directory in the ``models`` directory and in each model directory there are
three ``.py`` files:
* ``run.py`` is the file from which the model is run.  It contains the main timestepping loop, which allows a model to make extra adjustments after the main solve step has been completed,
e.g. deforming the grid.
* ``setup.py`` contains the code required to create the initial conditions of the model.  It is also where all model parameters are set.
* ``visualisations.py`` contains the model's version of the function ``makePlots``, which is called in ``run.py`` to make plots.  This is also where any plotting routines 
specific to that model can be defined. 

Each model directory also contains a ``material_properties.txt`` file.  This contains the material parameters for all materials required by the model.

More details on the contents of these files can by found in :ref:`creating-a-model`. 

jit compilation
---------------
The code uses just-in-time (jit) compilation from the numba package.  This compiles functions on their first usage, and then reuses this compiled code for all subsequent calls of that function, improving the performance of the code dramatically (you'll notice that the first timestep is much slower than the following ones as this includes all the compile time).  

If you wish to disable jit e.g. for debugging, change the environment variable ``NUMBA_DISABLE_JIT`` to 1 (there is a line after the import statements in ``main.py`` that can be used for this) and comment out the ``@jitclass`` decorators on all the class definitions.  To switch it back on, you'll need to restart you python kernel.

Visualisation
-------------

The ``output/visualisation.py`` file contains some common functions for plotting.  To make plots with variants of these functions, you can copy them into the model-specific ``visualisations.py`` file and edit them there, then call your new version in the ``makePlots`` function.  :py:class:`dataStructures.Grid` shows the available grid variables for plotting.

.. note:: The colorbar limits and plotting areas are set manually in the plotting functions, you should adjust these to fit your simulation.

