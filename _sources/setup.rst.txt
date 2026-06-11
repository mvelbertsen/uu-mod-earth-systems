.. _creating-a-model:

Creating a new model
====================

To create a new model, first create a new directory in the models directory, with a name that describes what your model will be.  
This directory needs to contain a version of ``run.py``, ``setup.py``, ``visualisation.py`` and a ``material_properties.txt``.  

The ``setup.py`` file should contain two main functions:

:py:func:`models.simpleStokes.setup.initializeModel`

which is used to create the grid, intantiate the parameters and set boundary conditions and required parameters for a simulation.  It then calls a second function,

:py:func:`models.simpleStokes.setup.initialize_markers`

which creates and distributes the markers across the simulation domain.  This function is where the initial material properties and temperatures are assigned to the markers based on position, setting up the simulation's initial state.
 
Finally, a function is required to set the initial grid up.  

:py:func:`models.common.uniformGrid`

can be called to create a uniformly spaced, static grid.  If you wish to set up a different kind of grid, you should implement a function which does this in ``setup.py``.  If the grid should also change in time, then this grid function should also be called to update the grid in ``run.py``.

The ``Subduction`` model shows an example of a non-uniform grid, it sets up a static Swiss Cross grid.

For a further example, the ``lithosphereExtension`` model uses a grid with a fixed high resolution area in the upper central region of the domain, with an increasing grid spacing moving outward.  
This version also expands in the x direction over time, and so is called to update the grid in ``run.py``.  

For your own simulation, you can either copy one of these geometries or create your own implementation.  Each version requires certain parameters to be defined in the ``Parameters`` object.

.. note:: The easiest way to create a new setup is to copy an existing one and edit it! 

Parameters
----------

There should also be a copy of the Parameters object in ``setup.py``.  This defines all the numerical and physical parameters required for your simulation, there is a base set of parameters that are required for any simulation, which are shown in the example implementation.  You can also add further parameters, by copying the Parameters definition to your setup.py and adding extra attributes, remembering to also add their types to the spec_par list.

For a basic example of the required parameters see :py:class:`models.simpleStokes.setup.Parameters`

.. note:: This cannot be done by creating a class that inherits from the example as ``jitclass`` does not support inheritance.

Material Properties
-------------------
This text file should contain the required properties for each different material in your model.  Each row should contain the following values in the order they are listed in the table:

+------------------------+-----------------------------------------+-----------+
| Parameter              | Description                             | Optional? |
+========================+=========================================+===========+
| density, :math:`\rho`  | basic density of the material           | no        |
+------------------------+-----------------------------------------+-----------+
| thermal expansion      | gives change in density due to thermal  | yes       |
| coefficient :math:`c_T`| expansion as: :math:`\rho (1 - c_T T)`  |           |
+------------------------+-----------------------------------------+-----------+
| Compressibility        | gives change in density due to pressure | yes       |
| coefficient :math:`c_P`| as :math:`\rho (1 + c_P P)`             |           |
+------------------------+-----------------------------------------+-----------+
| Viscosity model        | flag which specifies the viscosity      | no        |
| choice                 | model used:                             |           |
|                        | 0 = constant, 1 = power law             |           |
+------------------------+-----------------------------------------+-----------+
| Constant viscosity     | If a constant viscosity is chosen, this | depends   |
|                        | contains the value of that viscosity    | on flag   |
+------------------------+-----------------------------------------+-----------+
| Power law viscosity,   | The material constant for the           | depends   |
| :math:`A_d` parameter  | power law viscosity model               | on flag   |
|                        | :math:`A_d\sigma^n exp(-(E_a+V_a P)/RT)`|           |
+------------------------+-----------------------------------------+-----------+
| Power law viscosity,   | The stress exponent for the             | depends   |
| :math:`n` parameter    | power law viscosity model               | on flag   |
|                        | :math:`A_d\sigma^n exp(-(E_a+V_a P)/RT)`|           |
+------------------------+-----------------------------------------+-----------+
| Power law viscosity,   | Experimentally determined rheological   | depends   |
| :math:`E_a` parameter  | parameter for power law viscosity model | on flag   |
|                        | :math:`A_d\sigma^n exp(-(E_a+V_a P)/RT)`|           |
+------------------------+-----------------------------------------+-----------+
| Power law viscosity,   | Experimentally determined rheological   | depends   |
| :math:`V_a` parameter  | parameter for power law viscosity model | on flag   |
|                        | :math:`A_d\sigma^n exp(-(E_a+V_a P)/RT)`|           |
+------------------------+-----------------------------------------+-----------+
| Shear modulus          | The shear modulus of the material       | no        |
+------------------------+-----------------------------------------+-----------+
| Plasticity, cohesion   | The cohesion used when the strain is    | yes       |
| pre strain weakening   | below the lower strain threshold        |           |
| :math:`C_0`            | :math:`\gamma_0`                        |           |
+------------------------+-----------------------------------------+-----------+
| Plasticity, cohesion   | The cohesion used when the strain is    | yes       |
| post strain weakening  | above the upper strain threshold        |           |
| :math:`C_1`            | :math:`\gamma_1`                        |           |
+------------------------+-----------------------------------------+-----------+
| Plasticity, internal   | The internal friction used when the     | yes       |
| friction pre strain    | strain is below the lower strain        |           |
| weakening              | threshold  :math:`\gamma_0`             |           |
+------------------------+-----------------------------------------+-----------+
| Plasticity, internal   | The internal friction used when the     | yes       |
| friction post strain   | strain is above the upper strain        |           |
| weakening              | threshold  :math:`\gamma_1`             |           |
+------------------------+-----------------------------------------+-----------+
| Plasticity, lower      | The strain threshold below which no     | yes       |
| strain threshold,      | strain weaking is applied               |           |
| :math:`\gamma_0`       |                                         |           |
+------------------------+-----------------------------------------+-----------+
| Plasticity, upper      | The strain threshold above which full   | yes       |
| strain threshold,      | strain weaking is applied, in between   |           |
| :math:`\gamma_1`       | a combination of the parameters is used |           |
+------------------------+-----------------------------------------+-----------+
| Specific heat capacity | The specific heat capacity at constant  | no        |
| :math:`C_p`            | pressure                                |           |
+------------------------+-----------------------------------------+-----------+
| Thermal conductivity   | The constant thermal conductivity       | no        |
| :math:`k_T`            |                                         |           |
+------------------------+-----------------------------------------+-----------+
| Thermal conductivity   | The coefficient of temperature          | yes       |
| temperature coefficient| dependence for the thermal conductivity |           |
| :math:`a`              | :math:`k_T + a/(T+77)`                  |           |
+------------------------+-----------------------------------------+-----------+
| Radiogenic heating     | Heat production from radiogenic sources | yes       |
| term (W/m**3)          |                                         |           |
+------------------------+-----------------------------------------+-----------+

for values which are optional, 0 should be entered if they are not required/in use.  This must be done consistently, for example, all of the plasticity parameters should be zeroed if plasticity is not in use.


Boundary Conditions
-------------------

The boundary conditions for the simulation are controlled by the :py:class:`solver.physics.boundaryConditions.BCs` class.
This stores the arrays for both the velocity and temperature boundary conditions on each wall, as well as the pressure in the first node, and the options for including a fixed velocity wall inside the simulation.

It also includes helper functions that can be used to easily set some common boundary conditions.

Velocity
^^^^^^^^

The helper functions in the ``BCs`` class allow you to set the following conditions: "free slip", "no slip" and "prescribed parallel velocity" by using ::
    
    BC.set_top_BC("free slip")

(replacing ``top`` with any other direction for other walls).  The full list of helper functions and options can be found here: :py:class:`model.solver.physics.boundaryConditions.BCs`.

If you wish to use another kind of boundary condition, then you must manually set the boundary arrays.  

The implementation is detailed here, and the ``LithosphereExtension`` model's ``initializeModel`` also shows an example of setting a non-standard boundary condition.

The boundary conditions applied at each wall for velocities are specified in 4 arrays: ``B_top``, ``B_bottom``, ``B_left`` and ``B_right``, where top/bottom refers to the y-direction and left/right the x.  Each contain 4 columns with xnum/ynum elements for the top and bottom / left and right arrays repectively.  The first two columns are the :math:`v_x` conditions and the second two are the :math:`v_y` conditions.  The values in the columns set the values for the i/jth (for the x/y directions respectively) 'ghost node' velocities as:

``vx[0,j] = B_bottom[j,0] + vx[1,j]*B_bottom[j,1]``

and for the :math:`v_y` condition,

``vy[0,j] = B_bottom[j,2] + vy[1,j]*B_bottom[j,3]``

The table below shows how to implement several common boundary conditions in this structure, note that the single row shown is representing the same pattern in all rows of the array:

+------------------------+----------------------------------------------+
| Boundary condition     | Array structure (showing a single row)       |
+========================+==============================================+
| No slip                | ``[0, 0, 0, 0]``                             |
+------------------------+----------------------------------------------+
| Free slip              | ``[0, 1, 0, 0]`` for top / bottom,           |
|                        | ``[0, 0, 0, 1]`` for left / right            |
+------------------------+----------------------------------------------+
| Grid deformation       | ``[0, 1, -v/xsize*ysize, 0]`` for top/bottom |
| spreading in x         | ``[(-/+)v/2, 0, 0, 1]`` for left/right       |
+------------------------+----------------------------------------------+
| Prescribed inflow      | ``[v, 0, 0, 0]`` for top / bottom,           |
| parallel to boundary   | ``[0, 0, v, 0]`` for left / right            |
+------------------------+----------------------------------------------+

Temperature
^^^^^^^^^^^
The conditions "insulating" and "fixed T" can be set using the ``BCs`` class' helper functions as::

    BCs.set_top_T_BC("insulating")

Again, other conditions require you to manually set the boundary array values.  
A similar structure as in the velocity case is used for the temperature boundary conditions.  These are again specified in 4 arrays: ``BT_top``,``BT_bottom``, ``BT_left`` and ``BT_right``.  Each array contains two columns that are used to calculate the ghost node temperature as:

``T[0,j] = BT_top[0] + BT_top[1]*T[1,j]``

Some common boundary condition types (these are already implemented by helper functions) can be formulated in this structure as:

+------------------------+------------------------+
| Boundary condition     | Array structure        |
|                        | (showing a single row) |
+========================+========================+
| Insulating             | ``[0, 1]``             |
+------------------------+------------------------+
| Prescribed temperature | ``[T, 0]``             |
+------------------------+------------------------+

Internal velocity Boundary
--------------------------
There is also an option to include a 'mobile wall', which is a vertical wall within the simulation domain on which a :math:`x` and/or :math:`y` velocity can be fixed.  The :math:`x` and :math:`y` velocities can be fixed on the same or separate vertical lines.  This is implemented using the ``B_intern`` array, which has the following format:

+-------+-------------------------------+------------+
| index | description                   | value when | 
|       |                               | not in use |
+=======+===============================+============+
| 0     | x index of the wall on        | -1         |
|       | which the x velocity is fixed |            | 
+-------+-------------------------------+------------+
| 1     | min y-index of the wall       | 0          |
+-------+-------------------------------+------------+
| 2     | max y-index of the wall       | 0          |
+-------+-------------------------------+------------+
| 3     | x-velocity on the wall        | 0          |
|       | described by elements 0-2     |            |
+-------+-------------------------------+------------+
| 4     | x index of the wall on        | -1         |
|       | which the y velocity is fixed |            | 
+-------+-------------------------------+------------+
| 5     | min y-index of the wall       | 0          |
+-------+-------------------------------+------------+
| 6     | max y-index of the wall       | 0          |
+-------+-------------------------------+------------+
| 7     | y-velocity on the wall        | 0          |
|       | described by elements 4-6     |            | 
+-------+-------------------------------+------------+ 

A simple example which shows the effect of this internal wall is in ``models/internalVelocityExample``.

Initial Conditions
------------------
The initial conditions for the simulation are set by applying material IDs and temperatures to the markers in :py:func:`models.simpleStokes.setup.initialize_markers`.

Within this function, the markers are distributed evenly across the domain with a small random displacement.  The user can then assign them a material type/ID and a temperature based on their position.  

For example, in the ``lithosphereExtension`` model, the different materials are assigned using depth, to give the layers of the crust and upper mantle.


