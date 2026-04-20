#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple intrusion with visc, T,  density contrast
"""
import numpy as np

from solver.dataStructures import Markers, Grid, Materials, ViscBox
from solver.physics.boundaryConditions import BCs
from models.common import uniformGrid

from numba import float64, int64, typeof
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
    BC : BCs Class
        Object containing all boundary condition arrays for velocity, pressure and temperature 

    '''
    
    # instantiate a pre-populated parameters object
    params = Parameters()

    # set resolution
    xnum = 31
    ynum = 21
    

    # instantiate/load material properties object
    matData = np.loadtxt('./material_properties_simple.txt', delimiter=",")
    materials = Materials(matData) 

    params.save_fig = 2
    params.ntstp_max = 10


    ###########################################################################    
    # Boundary conditions
    BC = BCs(xnum, ynum)
    
    # pressure BCs
    BC.P_first[1] = 1e5

    # velocity BCs
    BC.set_top_BC("free slip")
    BC.set_bottom_BC("free slip")
    BC.set_left_BC("free slip")
    BC.set_right_BC("free slip")

    
    # temperature BCs
    # upper and lower  - insulating
    BC.set_top_T_BC("insulating")
    BC.set_bottom_T_BC("insulating")

    # left and right = insulating
    BC.set_left_T_BC("insulating")
    BC.set_right_T_BC("insulating")

    ###########################################################################
    # create grid object
    grid = Grid(xnum, ynum)

    # define grid points for evenly spaced grid
    uniformGrid(params, grid)

    ############################################################################
    # create markers object
    mnumx = 400
    mnumy = 300
    markers = Markers(mnumx, mnumy)

    # initialize markers
    initialize_markers(markers, materials, params)
    
    return params, grid, materials, markers, BC
           
           

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
            # inside the intrusion
            if (markers.y[mm] > params.ysize*0.4 and markers.y[mm] < params.ysize*0.6 and markers.x[mm] > params.xsize*0.4 and markers.x[mm] < params.xsize*0.6):
                markers.id[mm] = 0
                markers.T[mm] = 1300
            else:
                markers.id[mm] = 1
                markers.T[mm] = 1000
                
            
            # update marker index
            mm +=1


            
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
    ('save_output', int64),
    ('save_fig', int64),
    ('output_name', unicode_type),
    ('output_path', unicode_type),
    ('viscbox', typeof(ViscBox(0)))
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
    output_path : STR
        The output path where the result should be written to specified relative to the run.py file's location.
    viscbox : ViscBox Class
        Object containing parameters for controlling the optional high viscosity box.
    
    '''
    
    
    # creates parameters object
    def __init__(self):
        '''
        Constructor for the parameters class

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
        self.xsize = 1.0e5                      # physical x-size of model, m
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
        
        
        # output options
        self.save_output = 50                        # number of steps between output files
        self.save_fig = 20                            # number of steps between figure output
        self.output_name = "simpleStokesAndTemp"       # name of the folder to write data to (within the main figures directory)
        self.output_path = "../../Results/figures"
        
        self.viscbox = ViscBox(0)               # high viscosity box, switched off
