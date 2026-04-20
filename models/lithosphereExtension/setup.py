#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup for a lithospheric extension model, the default setup for the original code.

"""

import numpy as np

from numba import jit, float64, int64, typeof
from numba.types import unicode_type
from numba.experimental import jitclass

from solver.dataStructures import Markers, Grid, Materials, ViscBox
from solver.physics.boundaryConditions import BCs


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
    xnum = 161
    ynum = 61


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

    # manually set the bottom, left and right boundaries to include the grid velocity
    BC.B_bottom[:,1] = 1
    BC.B_bottom[:,2] = -params.v_ext/params.xsize * params.ysize

    BC.B_left[:,0] = -params.v_ext/2
    BC.B_left[:,3] = 1

    BC.B_right[:,0] = params.v_ext/2
    BC.B_right[:,3] = 1

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

    # define grid points for (potentially) unevenly spaced grid
    updateGrid(params, grid, 0, 0.0, BC.B_bottom)

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
            markers.T[mm] = params.T_bot - dtdy*(params.ysize - markers.y[mm])
            
            # if in the air
            if (markers.id[mm]==0):
                # cons T
                markers.T[mm] = params.T_top
            
            # linear continental geotherm
            y_asth = 97000
            T_asth = params.T_bot - dtdy*(params.ysize - y_asth)
            if (markers.y[mm] > 7000 and markers.y[mm]<y_asth):
                markers.T[mm] = params.T_top + (T_asth - params.T_top)*(markers.y[mm] - 7000)/(y_asth - 7000)
            
            # seed to start localisation in center of model
            # using a thermal perturbation
            dx = markers.x[mm] - params.xsize/2
            dy = markers.y[mm] - 40000
            radius = np.sqrt(dx**2 + dy**2)
            if (radius<50000):
                markers.T[mm] = markers.T[mm] + 60*(1-radius/50000)
            
            
            # update marker index
            mm +=1
    
    


def updateGrid(params, grid, t_curr, timestep, BC_bot):
    '''
    Calculates the new grid point spacings based on the current xsize and ysize.
    This version contructs a non-uniform grid with a central-upper high resolution region
    and decreasing resolution outward from this.

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
        return 0
    
    xnum = grid.xnum
    ynum = grid.ynum
    
    # pull out the required parameters from params object
    bx = params.bx
    by = params.by
    Nx = params.Nx
    Ny = params.Ny
    
    non_uni_xsize = params.non_uni_xsize

    
    ###############################################################################
    # Horizontal grid

    if (t_curr==0):
        # set the points in the high res area
        # this only needs doing on the initial setup
        grid.x[Nx] = non_uni_xsize 
        for i in range(Nx+1,xnum-Nx):  #xnum
            grid.x[i] = grid.x[i-1] + bx
            
        # size of the non-uniform region 
        D = params.xsize - grid.x[xnum-Nx-1]

    else:
        
        # update grid positions based on extension
        params.ysize += -params.v_ext/params.xsize*params.ysize*timestep
        params.xsize += params.v_ext*timestep
        
        # set the new position of the first node,
        # and the size of the non-uniform region
        grid.x[0] = grid.x[int(xnum/2)] - params.xsize/2
        D = params.xsize/2 - (grid.x[xnum-Nx-1] - grid.x[int(xnum/2)])
        
        # if we have changing grid, we also need to update bottom BC
        if (abs(params.v_ext)>0):
            BC_bot[:,2] = -params.v_ext/params.xsize*params.ysize
        
    
    # define factor of grid spacing to increase to the right of high res area
    # need to only do this for non-uniform def, otherwise div by 0!
    if (Nx > 0):
        F = 1.1
        # iteratively solve for F
        for i in range(0,200):
            F = (1 + D/bx*(1 - 1/F))**(1/Nx)
    
        # define grid points to the right of the high-res region
        for i in range(xnum-Nx, xnum):
            grid.x[i] = grid.x[i-1] + bx*F**(i-(xnum-Nx-1))
        
        if (t_curr==0):
            grid.x[xnum-1] = params.xsize
    
        # now do the same going leftward
        D = grid.x[Nx] - grid.x[0] # think this should still work for inital case?
    
        F = 1.1
        for i in range(0,100):
            F = (1 + D/bx*(1 - 1/F))**(1/Nx)
    
        # set the points left of the high res region
        for i in range(1,Nx):
            grid.x[i] = grid.x[i-1] + bx*F**(Nx+1-i)
        
    
    ###########################################################################
    # Vertical grid
    # one-sided, there is high resolution at the top of the grid and then a decreasing region below

    # set the high resolution area, assumes y[0] = 0
    for i in range(1,ynum-Ny):
        grid.y[i] = grid.y[i-1] + by
      
    
    if (Ny > 0):
        # size of the non-uniform regions
        D = params.ysize - grid.y[ynum-Ny-1]
       
        # solve iteratively for scaling factor
        F = 1.1
        for i in range(0,100):
            F = (1 + D/by*(1 - 1/F))**(1/Ny)
            # set the grid points below the high-res region
        for i in range(ynum-Ny, ynum):
            grid.y[i] = grid.y[i-1] + by*F**(i-(ynum-Ny-1))
           
        # fix the end position if this is the first step
        if (t_curr==0):
            grid.y[ynum-1] = params.ysize



    

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
        self.gx = 0.0                           # x-direction gravitational acc
        self.gy = 9.81                          # y-direction gravitational acc
        self.Rgas = 8.314                       # gas constant
        
        # physical model setup
        self.xsize = 400000
        self.ysize = 300000
        self.T_min = 273                        # temperature at the top face of the model (K)
        
        
        # viscosity model
        self.eta_min = 1e18                     # minimum viscosity
        self.eta_max = 1e25                     # maximum viscosity
        self.stress_min = 1e4                   # minimum stress
        self.eta_wt = 0                         # viscosity weighting, for (old?) visco-plastic model
        self.max_pow_law = 150                  # maximum power law exponent in visc model
        
        self.v_ext = 2.0/(100*365.25*24*3600)   # extension velocity of the grid (cm/yr)
        
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
        
        # output options
        self.save_output = 50                        # number of steps between output files
        self.save_fig = 10                           # number of steps between figure output
        self.output_name = "lithosphereExtension"
        self.output_path = "../../Results/figures"
        
                
        ########################################################################
        # params specific only to this setup
        
        # grid spacing params
        
        self.bx = 2000                          # x-grid spacing in high res area
        self.by = 2000                          # y-grid spacing in high res area
        self.Nx = 30                            # number of unevenly spaced grid points either side of high res zone
        self.Ny = 20                            # number of unvenly spaced grid points below high res zone
        self.non_uni_xsize = 100000             # physical x-size of non-uniform grid region
        self.const = 0                          # flag which determines whether grid remains constant or not
        

        self.viscbox = ViscBox(0)

        self.T_top = self.T_min                 # temperature at the top face of the model (K)
        self.T_bot = 1750                       # temperature at the bottom face of the model (K)








    
