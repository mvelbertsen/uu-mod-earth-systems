Running your model
==================

To run a model, simply check that you have imported the ``initializeModel`` for your chosen simulation at the top of the ``main.py`` fil, then run the ``main.py`` script!

jit compilation
---------------
The code uses just-in-time (jit) compilation from the numba package.  This compiles functions on their first usage, and then reuses this compiled code for all subsequent calls of that function, improving the performance of the code dramatically (you'll notice that the first timestep is much slower than the following ones as this includes all the compile time).  

If you wish to disable jit e.g. for debugging, change the environment variable ``NUMBA_DISABLE_JIT`` to 1 (there is a line after the import statements in ``main.py`` that can be used for this) and comment out the ``@jitclass`` decorators on all the class definitions.  To switch it back on, you'll need to restart you python kernel.

Visualisation
-------------

The ``visualisation.py`` file contains some basic functions for plotting.  To change what variables are plotted, simply change the grid variable in the plot command within these functions to your chosen variable.  :py:class:`dataStructures.Grid` shows the available grid variables for plotting.



