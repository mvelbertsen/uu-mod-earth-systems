#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup for a lithospheric extension model, the default setup for the original code.

"""

import numpy as np
import pathlib
from numba import jit, float64, int64
from numba.experimental import jitclass

from dataStructures import Markers, Grid, Materials
from physics.grid_fns import gridSpacings



def initialize_markers(markers, materials, params, xsize, ysize):
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
    xsize : FLOAT
        Physical x-size of the simulation domain.
    ysize : FLOAT
        Physical y-size of the simulation domain.

    Returns
    -------
    None.

    '''
    
    np.random.seed(1337)
    # set the rough grid spacing for the markers
    mxstp = xsize/markers.xnum
    mystp = ysize/markers.ynum
    
    # marker number index
    mm = 0
    
    for j in range(0,markers.xnum):
        for i in range(0,markers.ynum):
            
            # set coordinates as grid + small random displacement
            markers.x[mm] = (j + np.random.random())*mxstp
            markers.y[mm] = (i + np.random.random())*mystp
            
            # now define rock type based on location, to give our initial setup
            
            # asthenosphere
            markers.id[mm] = 5
            
            # sticky air
            if (markers.y[mm] <= 10000):
                markers.id[mm] = 0
            
            # continental lithosphere
            if (markers.y[mm]>7000 and markers.y[mm]<= 12000):
                markers.id[mm] = 7
            if (markers.y[mm]>12000 and markers.y[mm]<=17000):
                markers.id[mm] = 8
            if (markers.y[mm]>17000 and markers.y[mm]<= 22000):
                markers.id[mm] = 7
            # lower cont crust
            if (markers.y[mm]>22000 and markers.y[mm]<= 27000):
                markers.id[mm] = 9
            if (markers.y[mm]>27000 and markers.y[mm]<= 32000):
                markers.id[mm] = 10
            if (markers.y[mm]>32000 and markers.y[mm]<= 37000):
                markers.id[mm] = 9
            # lithospheric mantle, default is 5, we add stripes of 4
            if (markers.y[mm]>42000 and markers.y[mm]<= 47000):
                markers.id[mm] = 4
            if (markers.y[mm]>52000 and markers.y[mm]<= 57000):
                markers.id[mm] = 4
            if (markers.y[mm]>62000 and markers.y[mm]<= 67000):
                markers.id[mm] = 4
            if (markers.y[mm]>72000 and markers.y[mm]<= 77000):
                markers.id[mm] = 4
            if (markers.y[mm]>82000 and markers.y[mm]<= 87000):
                markers.id[mm] = 4
            if (markers.y[mm]>92000 and markers.y[mm]<= 97000):
                markers.id[mm] = 4
            
            # initial temperature structure
            # adiabatic T gradient in asthenosphere
            dtdy = 0.5/1000 # K/m
            markers.T[mm] = params.T_bot - dtdy*(ysize - markers.y[mm])
            
            # if in the air
            if (markers.id[mm]==0):
                # cons T
                markers.T[mm] = params.T_top
            
            # linear continental geotherm
            y_asth = 97000
            T_asth = params.T_bot - dtdy*(ysize - y_asth)
            if (markers.y[mm] > 7000 and markers.y[mm]<y_asth):
                markers.T[mm] = params.T_top + (T_asth - params.T_top)*(markers.y[mm] - 7000)/(y_asth - 7000)
            
            # seed to start localisation in center of model
            # using a thermal perturbation
            dx = markers.x[mm] - xsize/2
            dy = markers.y[mm] - 40000
            radius = np.sqrt(dx**2 + dy**2)
            if (radius<50000):
                markers.T[mm] = markers.T[mm] + 60*(1-radius/50000)
            
            
            # update marker index
            mm +=1
    
    
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
    xsize : INT
        x-resolution of model domain.
    ysize : INT
        y-resolution of model domain.
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
    
    
    # instantiate a pre-populated parameters object, using the derived class
    params = Parameters()

    # additional model options 
    # initial system size
    xsize0 = 400000
    ysize0 = 300000
    
    xsize = xsize0
    ysize = ysize0

    # set resolution
    xnum = 161
    ynum = 61


    # instantiate/load material properties object
    # file path must be from top directory (as that is where the fn is called from!)
    matData = np.loadtxt('./models/lithosphereExtension/material_properties.txt', delimiter=",")
    materials = Materials(matData)

    # create directories for output of figures and data (not atm)
    pathlib.Path('./Figures').mkdir(exist_ok=True)
    pathlib.Path('./Output').mkdir(exist_ok=True)

    ###########################################################################    
    # Boundary conditions
    # pressure BCs
    P_first = np.array([0,1e5])

    # velocity BCs
    B_top = np.zeros((xnum+1,4))
    B_top[:,1] = 1

    B_bottom = np.zeros((xnum+1,4))
    B_bottom[:,1] = 1
    B_bottom[:,2] = -params.v_ext/xsize * ysize

    B_left = np.zeros((ynum+1,4))
    B_left[:,0] = -params.v_ext/2
    B_left[:,3] = 1

    B_right = np.zeros((ynum+1,4))
    B_right[:,0] = params.v_ext/2
    B_right[:,3] = 1

    # optional internal boundary, switched off
    B_intern = np.zeros(8)
    B_intern[0] = -1
    B_intern[4] = -1
    
    # temperature BCs
    BT_top = np.zeros((xnum, 2))
    BT_bottom = np.zeros((xnum, 2))
    BT_left = np.zeros((ynum, 2))
    BT_right = np.zeros((ynum, 2))

    # upper and lower  = fixed T
    BT_top[:,0] = params.T_top
    BT_bottom[:,0] = params.T_bot

    # left and right = insulating
    BT_left[:,1] = 1
    BT_right[:,1] = 1

    ###########################################################################
    # create grid object
    grid = Grid(xnum, ynum)

    # define grid points for (potentially) unevenly spaced grid
    gridSpacings(params.bx, params.by, params.Nx, params.Ny, params.non_uni_xsize, xsize, ysize, grid, 0)

    ############################################################################
    # create markers object
    mnumx = 400
    mnumy = 300
    markers = Markers(mnumx, mnumy)

    # initialize markers
    initialize_markers(markers, materials, params, xsize, ysize)
    
    return params, grid, materials, markers, xsize, ysize, P_first, B_top, B_bottom,\
           B_left, B_right, B_intern, BT_top, BT_bottom, BT_left, BT_right
    
    

spec_par = [
    ('gx', float64),
    ('gy', float64),
    ('Rgas', float64),
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
    ('bx', float64),
    ('by', float64),
    ('Nx', int64),
    ('Ny', int64),
    ('non_uni_xsize', float64),
    ('save_output', int64),
    ('save_fig', int64),
    ('T_top', float64),
    ('T_bot', float64),
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
    bx: FLOAT
        x-grid spacing in the high resolution region of the grid.  For uniform grid
        this should be xsize/(xnum-1).
    by: FLOAT
        y-grid spacing in the high resolution region of the grid.  For uniform grid
        this should be ysize/(ynum-1).
    Nx: INT
        Number of uneven grid cells either side of the central high resolution 
        region in the x-direction, for uniform grid this should be 0.
    Ny: INT
        Number of uneven grid cells below the high resolution 
        region in the y-direction, for uniform grid this should be 0.
    non_uni_size: FLOAT
        Physical size of the non-uniform grid region in the x-direction.  For 
        uniform grid this should be 0.
    save_output : INT
        Number of steps between output, not currently implemented.
    save_fig : INT
        Number of steps between plotting of figures.
    T_top : FLOAT
        Temperature at the top of the simulation domain.
    T_bot : FLOAT
        Temperature at the bottom of the simulation domain.
    
    
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
        self.T_min = 273                        # temperature at the top face of the model (K)
        
        self.v_ext = 2.0/(100*365.25*24*3600)   # extension velocity of the grid (cm/yr)
        
        # viscosity model
        self.eta_min = 1e18                     # minimum viscosity
        self.eta_max = 1e25                     # maximum viscosity
        self.stress_min = 1e4                   # minimum stress
        self.eta_wt = 0                         # viscosity weighting, for (old?) visco-plastic model
        self.max_pow_law = 150                  # maximum power law exponent in visc model
        
        
        # timestepping
        self.t_end = 72e3/self.v_ext            # end time
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
        self.bx = 2000                          # x-grid spacing in high res area
        self.by = 2000                          # y-grid spacing in high res area
        self.Nx = 30                            # number of unevenly spaced grid points either side of high res zone
        self.Ny = 20                            # number of unvenly spaced grid points below high res zone
        self.non_uni_xsize = 100000             # physical x-size of non-uniform grid region
        
        # output options
        self.save_output = 50                        # number of steps between output files
        self.save_fig = 20                            # number of steps between figure output

        
        ########################################################################
        # params specific only to this setup
        self.T_top = self.T_min                 # temperature at the top face of the model (K)
        self.T_bot = 1750                       # temperature at the bottom face of the model (K)








    
