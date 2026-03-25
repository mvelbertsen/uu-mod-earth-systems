#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Stoke's flow test with constant visc, T and vertical density contrast
"""
import numpy as np

from solver.dataStructures import Markers, Grid, Materials
from numba import float64, int64
from numba.types import unicode_type
from numba.experimental import jitclass

def initializeModel():
    '''
    Sets up the initial state of the model, including BCs and output settings.

    Returns
    -------
    params : Parameters object
        Model physical and numerical parameters.
    grid : Grid object
        Initialised Grid object.
    materials : Materials
        Materials object initialsed with required material properties.
    markers : Markers
        Initialized markers object.
    P_first : ARRAY
        Array with 2 entries, specifying pressure BC.
    B_top : ARRAY
        Boundary conditions at the top of the grid. Array has 4 columns, 
        values in each are defined as: 
        vx[0,j] = B_top[j,0] + vx[1,j]*B_top[j,1]
        vy[0,j] = B_top[j,2] + vy[1,j]*B_top[j,3]
    B_bottom : ARRAY
        Boundary conditions at the bottom of the grid. Array has 4 columns, 
        values in each are defined as: 
        vx[0,j] = B_bot[j,0] + vx[1,j]*B_bot[j,1]
        vy[0,j] = B_bot[j,2] + vy[1,j]*B_bot[j,3]
    B_left : ARRAY
        Boundary conditions at the left of the grid. Array has 4 columns, 
        values in each are defined as: 
        vx[0,i] = B_left[i,0] + vx[1,i]*B_left[i,1]
        vy[0,i] = B_left[j,2] + vy[1,i]*B_left[i,3]
    B_right : ARRAY
        Boundary conditions at the left of the grid. Array has 4 columns, 
        values in each are defined as: 
        vx[0,i] = B_right[i,0] + vx[1,i]*B_right[i,1]
        vy[0,i] = B_right[j,2] + vy[1,i]*B_right[i,3]
    B_intern : ARRAY
        Array defining optional internal boundary eg. moving wall. Format is:
        B_intern[0] = x-index of vx nodes with prescribed velocity (-1 is not in use)
        B_intern[1-2] = min/max y-index of the wall
        B_intern[3] = prescribed x-velocity value.
        B_intern[4] = x-index of vy nodes with prescribed velocity (-1 is not in use)
        B_intern[5-6] = min/max y-index of the wall
        B_intern[7] = prescribed y-velocity value.
    BT_top : ARRAY
        Top temperature BCs.  Array has 2 columns, values in each are defined as:
        T[i,j] = BT_top[0] + BT_top[1]*T[i+1,j]
    BT_bottom : ARRAY
        Bottom temperature BCs.  Array has 2 columns, values in each are defined as:
        T[i,j] = BT_bottom[0] + BT_bottom[1]*T[i-1,j]
    BT_left : ARRAY
        Left temperature BCs.  Array has 2 columns, values in each are defined as:
        T[i,j] = BT_left[0] + BT_left[1]*T[i,j+1]
    BT_right : ARRAY
        Right temperature BCs.  Array has 2 columns, values in each are defined as:
        T[i,j] = BT_right[0] + BT_right[1]*T[i,j-1]

    '''
    
    # instantiate a pre-populated parameters object
    params = Parameters()
    
    params.v_ext = 0.0


    # set resolution
    xnum = 31
    ynum = 21


    # instantiate/load material properties object
    matData = np.loadtxt('./material_properties_simple.txt', delimiter=",")
    materials = Materials(matData)

    # output options
    params.save_fig = 2
    params.ntstp_max = 20


    ###########################################################################    
    # Boundary conditions
    # pressure BCs
    P_first = np.array([0,1e5])

    # velocity BCs
    B_top = np.zeros([xnum+1,4])
    B_top[:,1] = 1

    B_bottom = np.zeros([xnum+1,4])
    B_bottom[:,1] = 1

    B_left = np.zeros([ynum+1,4])
    B_left[:,3] = 1

    B_right = np.zeros([ynum+1,4])
    B_right[:,3] = 1

    # optional internal boundary, switched off
    B_intern = np.zeros([8])
    B_intern[0] = -1
    B_intern[4] = -1
    
    # temperature BCs
    BT_top = np.zeros([xnum, 2])
    BT_bottom = np.zeros([xnum, 2])
    BT_left = np.zeros([ynum, 2])
    BT_right = np.zeros([ynum, 2])

    # upper and lower  - symmetry
    BT_top[:,1] = 1
    BT_bottom[:,1] = 1

    # left and right = symmetry/insulating? = -1?
    BT_left[:,1] = 1
    BT_right[:,1] = 1

    ###########################################################################
    # create grid object
    grid = Grid(xnum, ynum)
    

    # define grid points for (potentially) unevenly spaced grid
    updateGrid(params, grid, 0, params.tstp_max, B_bottom)

    ############################################################################
    # create markers object
    mnumx = 400
    mnumy = 300
    markers = Markers(mnumx, mnumy)

    # initialize markers
    initialize_markers(markers, materials, params)
    
    return params, grid, materials, markers, P_first, B_top, B_bottom,\
           B_left, B_right, B_intern, BT_top, BT_bottom, BT_left, BT_right
           
           

def initialize_markers(markers, materials, params):
    '''
    Initialize the positions, material ID and temperature of the markers.
    
    Parameters
    ----------
    markers : Markers Object
        An empty markers object, which will be filled by this function.
        materials : Materials object
        Materials object filled from file with the required materials.
    params : Parameters Object
        Contains the simulation parameters

    Returns
    -------
    None.

    '''
    
    # set the rough grid spacing for the markers
    mxstp = params.xsize/markers.xnum
    mystp = params.ysize/markers.ynum
    
    # marker number index
    mm = 0
    
    for j in range(0,markers.xnum):
        for i in range(0,markers.ynum):
            
            # set coordinates as grid + small random displacement
            markers.x[mm] = (j + np.random.random())*mxstp
            markers.y[mm] = (i + np.random.random())*mystp
            
            # now define rock type based on location, to give our initial setup
            # left-side
            if (markers.x[mm] <= params.xsize/2):
                markers.id[mm] = 0
                markers.T[mm] = 1000
            else:
                markers.id[mm] = 1
                markers.T[mm] = 1000
                
            
            # update marker index
            mm +=1


def updateGrid(params, grid, t_curr, timestep, BC_bot):
    '''
    Calculates the new grid point spacings based on the current xsize and ysize.

    Parameters
    ----------
    params : Parameters Class
        Parameters object containing all simulation parameters for the system.
    grid : OBJ
        The grid object into which the new node positions will be written.
    t_curr : FLOAT
        The current simulation time, to determine whether to set up grid from scratch
        or extend an existing one.
    timestep : FLOAT
        The current timestep size.
    BC_bot : ARRAY
        The boundary condition which should also be updated by whatever changes
        are made to the grid, in this case the bottom.

    Returns
    -------
    None.

    '''
    
    if (t_curr > 0 and params.const==1):
        # we don't need to recalculate the grid, return here!
        return
    
    xnum = grid.xnum
    ynum = grid.ynum
    
    dx = params.xsize/(xnum-1)
    dy = params.ysize/(ynum-1)
    
    
    # Simple, uniform grid
    if (t_curr == 0):
        
        # horizontal grid
        grid.x[0] = 0
        for i in range(1,xnum):
            grid.x[i] = grid.x[i-1] + dx
        
        # vertical grid
        grid.y[0] = 0
        for i in range(1,ynum):
            grid.y[i] = grid.y[i-1] + dy
            
    else:
        # update grid positions based on extension
        params.ysize += -params.v_ext/params.xsize*params.ysize*timestep
        params.xsize += params.v_ext*timestep
        
        # if we have changing grid, need to update bottom BC
        if (abs(params.v_ext)>0):
            BC_bot[:,2] = -params.v_ext/params.xsize*params.ysize
            BC_bot[:,3] = 0
    
    if (params.const==0):
        raise ValueError('Moving, uniform grid is not implemented!')
    
    

            
            
spec_par = [
    ('gx', float64),
    ('gy', float64),
    ('Rgas', float64),
    ('xsize', float64),
    ('ysize', float64),
    ('T_min', float64),
    ('v_ext', float64),
    ('eta_min', float64),
    ('eta_max', float64),
    ('stress_min', float64),
    ('eta_wt', float64),
    ('max_pow_law', float64),
    ('t_end', float64),
    ('ntstp_max', int64),
    ('Temp_stp_max', int64),
    ('tstp_max', float64),
    ('marker_max', float64),
    ('marker_sch', int64),
    ('movemode', int64),
    ('dsubgrid', float64),
    ('dsubgridT', float64),
    ('frict_yn', float64),
    ('adia_yn', float64),
    ('const', int64),
    ('save_output', int64),
    ('save_fig', int64),
    ('output_name', unicode_type)
]
@jitclass(spec_par)
class Parameters():
    '''
    Class which holds the values of various physical and numerical parameters
    required in the simulation.
    
    Attributes
    ----------
    gx : FLOAT
        x-direction component of gravitational acceleration.
    gy : FLOAT
        y-direction component of gravitational acceleration.
    Rgas : FLOAT
        Ideal gas constant.
    xsize : FLOAT
        physical x-size of the grid.
    ysize : FLOAT
        physical y-size of the grid.
    T_min : FLOAT
        Minimum allowed temperature.
    v_ext : FLOAT
        Grid extension velocity, for deforming grid.
    eta_min : FLOAT
        Minimum allowed viscosity.
    eta_max : FLOAT
        Maximum allowed viscosity.
    eta_wt : FLOAT
        Weighting value for a (potentially not in use) visco-plastic model.
    max_pow_law : FLOAT
        Maximum allowed exponent in the power law viscosity model.
    t_end : FLOAT
        Time at which to end the simulation (if number of tsteps is less than ntstp_max).
    ntstp_max : INT
        Maximum number of timesteps to take.
    Temp_stp_max : INT
        Maximum number of temperature sub-timesteps to take.
    tstp_max : FLOAT
        Maximum size of timestep.
    marker_max : FLOAT
        Maximum fraction of average grid cell that a marker can move per timestep.
    marker_sch : INT
        Choice of advection scheme for markers, 1 = Euler, 4 = RK4
    movemode : INT
        Choice of how velocities are calculated, 0 = Stokes, no others implemented at present.
    dsubgrid : FLOAT
        Subgrid stress coefficient.
    dsubgridT : FLOAT
        Subgrid temperature diffusion coefficient.
    frict_yn : FLOAT
        Flag to apply friction heating.
    adia_yn : FLOAT
        Flag to apply adiabatic heating.
    const : INT
        Flag which determines if the grid points are fixed during the simulation or not
    save_output : INT
        Number of steps between output, not currently implemented.
    save_fig : INT
        Number of steps between plotting of figures.
    output_name : STR
        The name of the folder to write the output/figures to.  This will be located
        in models/{chosen_model}/figures/output_name.
    
    
    '''
    
    
    # creates parameters object
    def __init__(self):
        '''
        Constructor for the parameters class
        
        Sets the default (lithsphere extension) values for all params

        Returns
        -------
        None.

        '''
        # main parameters, required by all simulations
        
        # physical constants
        self.gx = 0.0                           # x-direction gravitational acc
        self.gy = 9.81                          # y-direction gravitational acc
        self.Rgas = 8.314                       # gas constant
        
        # physical model setup
        self.xsize = 1e5                        # physical x-size of model, m
        self.ysize = 1.5e5                      # physical y-size of model, m
        self.T_min = 273                        # temperature at the top face of the model (K)
        
        self.v_ext = 0.0                        # extension velocity of the grid (cm/yr)
        
        # viscosity model
        self.eta_min = 1e18                     # minimum viscosity
        self.eta_max = 1e25                     # maximum viscosity
        self.stress_min = 1e4                   # minimum stress
        self.eta_wt = 0                         # viscosity weighting, for (old?) visco-plastic model
        self.max_pow_law = 150                  # maximum power law exponent in visc model
        
        
        # timestepping
        self.t_end = 3e6*(365.24*24*3600)       # end time
        self.ntstp_max = 360                    # maximum number of timesteps
        self.Temp_stp_max = 20                  # maximum number of temperature substeps
        
        self.tstp_max = 1e4*365.25*24*3600      # maximum timestep
        
        # marker options
        self.marker_max = 0.3                   # maximum marker movement per timestep (fraction of av. grid step)
        self.marker_sch = 4                     # marker scheme 0 = no movement, 1 = Euler, 4=RK4
        
        self.movemode = 0                       # velocity calculation 0 = momentum eqn, 1 = solid body (not working currently)
        
        # subgrid diffusion
        self.dsubgrid = 1                       # subgrid stress coeff (none if zero)
        self.dsubgridT = 1                      # subgrid diffusion coeff(none if zero)
        
        # switches for heating terms
        self.frict_yn = 1                       # use friction heating?
        self.adia_yn = 1                        # use adiabatic heating?
        
        # grid spacing params
        self.const = 1                          # flag which determines whether grid remains constant or not
        
        # output options
        
        self.save_output = 50                   # number of steps between output files
        self.save_fig = 20                      # number of steps between figure output
        self.output_name = "simpleStokes"       # name of the folder to write data to (within the main figures directory)
