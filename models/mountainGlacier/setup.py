#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup for a mountain glacier model, the default setup for the original code.

"""

import numpy as np
import matplotlib.pyplot as plt

from numba import jit, float64, int64, typeof
from numba.types import unicode_type
from numba.experimental import jitclass

from solver.dataStructures import Markers, Grid, Materials, ViscBox
from solver.physics.boundaryConditions import BCs
from models.common import uniformGrid



def mountain_slope_curve(x, params):
    '''
    Define bedrock slope as horizontal line.

    '''

    return params.ysize - 75


def glacier_surface_curve(x, params):
    '''
    Define glacier surface slope as concave down decreasing curve.

    '''

    return 3500/(4000**4)*x**4 + 137.5  + (-x*0.16 + 200)




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
    BC : BCs Class
        Object containing all boundary condition arrays for velocity, pressure and temperature.
        
    '''
    
    
    # instantiate a pre-populated parameters object, using the derived class
    params = Parameters()

    # set resolution
    xnum = 321   # 12.5 m
    ynum = 33    # 12.5 m


    # instantiate/load material properties object
    # file path must be from top directory (as that is where the fn is called from!)
    matData = np.loadtxt('./material_properties.txt', delimiter=",")
    materials = Materials(matData)


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
    # upper and lower  = fixed T
    BC.set_top_T_BC("fixed T", params.T_top)
    BC.set_bottom_T_BC("fixed T", params.T_bot)

    # left and right = insulating
    BC.set_left_T_BC("insulating")
    BC.set_right_T_BC("insulating")

    ###########################################################################
    # create grid object
    grid = Grid(xnum, ynum)

    # define grid points for uniform grid
    uniformGrid(params, grid)

    ############################################################################
    # create markers object
    mnumx = xnum*4
    mnumy = ynum*4
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
    
    np.random.seed(1337)
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
            # bedrock
            if markers.y[mm] >= mountain_slope_curve(markers.x[mm], params) or markers.y[mm] >= params.ysize-params.by:
                markers.id[mm] = 1
                markers.T[mm]  = 293   # 20 °C
            # Bump in the bedrock
            elif markers.x[mm] >= 1800 and markers.x[mm] <2200 and markers.y[mm] > -0.125*markers.x[mm]+550 or markers.x[mm] >=2200 and markers.x[mm] <2300 and markers.y[mm] > 275 or markers.x[mm] >=2300 and markers.x[mm]<2400 and markers.y[mm] > 0.25*markers.x[mm]-300:
                markers.id[mm] = 1
                markers.T[mm] = 293
            #Basal ice layer
            elif markers.y[mm] >= 300 and markers.y[mm] <= 325 or markers.x[mm] >=1800 and markers.x[mm] < 2200 and markers.y[mm] > -0.125*markers.x[mm]+537.5:
                    markers.id[mm] = 4
                    markers.T[mm] = 273-15
            elif markers.y[mm] >= glacier_surface_curve(markers.x[mm], params):
                if (markers.y[mm] > 180 and markers.y[mm] <= 190) or (markers.y[mm] > 200 and markers.y[mm] <= 210) or (markers.y[mm] > 220 and markers.y[mm] <= 230) or (markers.y[mm] > 240 and markers.y[mm] <= 250) or (markers.y[mm]> 260 and markers.y[mm] <= 270):
                    markers.id[mm] = 3
                    markers.T[mm] = 273-15
                else:
                    markers.id[mm] = 2
                    markers.T[mm]  = 273-15   # -15°C

            # air         
            else: 
                markers.id[mm] = 0 
                markers.T[mm] = 273+10   # 10 °C

            # update marker index
            mm +=1
    


    

spec_par = [
    ('gx', float64),
    ('gy', float64),
    ('Rgas', float64),
    ('xsize', float64),
    ('ysize', float64),
    ('T_min', float64),
    ('eta_min', float64),
    ('eta_max', float64),
    ('stress_min', float64),
    ('eta_wt', float64),
    ('max_pow_law', float64),
    ('v_ext', float64),
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
    ('bx', float64),
    ('by', float64),
    ('Nx', int64),
    ('Ny', int64),
    ('non_uni_xsize', float64),
    ('const', int64),
    ('viscbox', typeof(ViscBox(0))),
    ('T_top', float64),
    ('T_bot', float64)
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
    const : INT
        Flag which determines if the grid points are fixed during the simulation or not
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
        theta   = np.arctan(0.16/1.)            # standard
        self.gx = 9.81*np.sin(theta)            # x-direction gravitational acc
        self.gy = 9.81*np.cos(theta)            # y-direction gravitational acc
        self.Rgas = 8.314                       # gas constant
        
        # physical model setup
        self.xsize = 4000
        self.ysize = 700-300
        self.T_min = 273+10                     # temperature at the top face of the model (K)
        
        # viscosity model
        self.eta_min = 1e5                      # minimum viscosity
        self.eta_max = 1e25                     # maximum viscosity
        self.stress_min = 1e5                   # minimum stress
        self.eta_wt = 0                         # viscosity weighting, for (old?) visco-plastic model
        self.max_pow_law = 5                    # maximum power law exponent in visc model
        
        self.v_ext = 2.0/(100*365.25*24*3600)   # extension velocity of the grid (cm/yr)
        
        # timestepping
        self.t_end = 1000*365.25*24*3600*2        # end time
        self.ntstp_max = 360*2                    # maximum number of timesteps
        self.Temp_stp_max = 20                  # maximum number of temperature substeps
        
        self.tstp_max = 365.25*24*3600          # maximum timestep
        
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
        self.save_output = 50                   # number of steps between output files
        self.save_fig = 30                      # number of steps between figure output
        self.output_name = "mountainGlacier/00refcoolbumpvtest"
        self.output_path = "../../Results/figures"
        
                
        ########################################################################
        # params specific only to this setup
        
        # grid spacing params
        
        self.bx = self.xsize/(321-1)               # x-grid spacing in high res area
        self.by = self.ysize/(33-1)                # y-grid spacing in high res area
        self.Nx = 0                                # number of unevenly spaced grid points either side of high res zone
        self.Ny = 0                                # number of unvenly spaced grid points below high res zone
        self.non_uni_xsize = 0                     # physical x-size of non-uniform grid region
        self.const = 1                             # flag which determines whether grid remains constant or not

        self.viscbox = ViscBox(0)

        self.T_top = 273+10 #                      # temperature at the top face of the model (K)
        self.T_bot = 273+20 #                      # temperature at the bottom face of the model (K)