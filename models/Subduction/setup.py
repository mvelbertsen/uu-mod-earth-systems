#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File which contains the setup for a Subduction model

"""

import numpy as np
from math import erf


from numba import float64, int64, typeof
from numba.types import unicode_type
from numba.experimental import jitclass

from solver.dataStructures import Markers, Grid, Materials, ViscBox
from solver.physics.boundaryConditions import BCs


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

            # Sediments
            if (markers.y[mm] > 10000 and markers.y[mm] <= 11000):
                markers.id[mm] = 1
            # Basaltic crust 
            if (markers.y[mm] > 11000 and markers.y[mm] <= 13000):
                markers.id[mm] = 2
            # Gabbroic crust
            if (markers.y[mm] > 13000 and markers.y[mm] <= 18000):
                markers.id[mm] = 3
            # Lithospheric mantle
            if (markers.y[mm] > 18000 and markers.y[mm] < 90000):
                markers.id[mm] = 4
            # Continent
            # Upper continental crust
            if (markers.x[mm] < 300000 and markers.y[mm] > 7000 and markers.y[mm] < 27000):
                markers.id[mm] = 7
            if (markers.x[mm] >= 300000 and markers.x[mm] < 350000 and  markers.y[mm] > 7000 + (markers.x[mm]-300000)/50000*3000 and markers.y[mm] < 32000 - (markers.x[mm]-300000)/50000*14000):
                markers.id[mm] = 7
            # Used for visualization of the model
            if (markers.x[mm] < 300000 and markers.y[mm] > 11000 and markers.y[mm] < 15000):
                markers.id[mm] = 8
            if (markers.x[mm] < 300000 and markers.y[mm] > 19000 and markers.y[mm] < 23000):
                markers.id[mm] = 8
            # lower cont crust
            if (markers.x[mm] < 300000 and markers.y[mm] > 27000 and markers.y[mm] < 42000):
                markers.id[mm] = 9
            if (markers.x[mm] < 300000 and markers.y[mm] > 32000 and markers.y[mm] < 37000):
                markers.id[mm] = 10
            if (markers.x[mm] >= 300000 and markers.x[mm] < 350000 and markers.y[mm] > 27000 - (markers.x[mm]-300000)/50000*14000 and markers.y[mm] < 42000 - (markers.x[mm]-300000)/50000*24000):
                markers.id[mm] = 9
    	    # Weak zone in the mantle
            if (markers.y[mm] > 28000 and markers.y[mm] < 90000 and markers.x[mm] > 340000 - (markers.y[mm]-18000)/24000*50000 and markers.x[mm] < 360000 - (markers.y[mm]-18000)/24000*50000):
                markers.id[mm] = 6            
	         # Weak zone in the oceanic crust
            if (markers.y[mm] > 20000 and markers.y[mm] <= 18000 and markers.x[mm] > 340000 - (markers.y[mm]-18000)/24000*50000 and markers.x[mm] < 360000 - (markers.y[mm]-18000)/24000*155000):
                markers.id[mm] = 2
	        # Asthenosphere below the oceanic plate
            if (markers.y[mm] > 70000 and markers.y[mm] <= 100000 and markers.x[mm] > 510000 - (markers.y[mm]-8000)/24000*75000):
                markers.id[mm] = 5

            # initial temperature structure
            # adiabatic T gradient in asthenosphere
            dtdy = 0.5/1000 # K/m
            markers.T[mm] = params.T_bot - dtdy*(params.ysize - markers.y[mm])
            
            # if in the air
            if (markers.id[mm]==0):
                # cons T
                markers.T[mm] = params.T_top
            
	        # Oceanic geotherm
            age = 3e7*(365.25*24*3600)      #Oceanic plate age, s
            y_asth = 92000   	           #Bottom of the lithosphere
            T_asth = params.T_bot - dtdy*(params.ysize - y_asth)  # T of astenosphere at y=yast
            kappa = 1e-6		     # Thermal diffusivity of the mantle, m^2/s
    
	        # After Turcotte & Schubert (2002) 
            if (markers.x[mm] > 350000 and markers.y[mm] > 10000 and markers.y[mm] < y_asth):  # markers.y was creater than 10000
                # T difference at the bottom of the oceanic plate
                dt = -(params.T_top - T_asth)*(1-erf((y_asth-10000)/2/(kappa*age)**0.5))
                markers.T[mm] = T_asth+dt+(params.T_top-T_asth-dt)*(1-erf((markers.y[mm]-10000)/2/(kappa*age)**0.5))

            # linear continental geotherm
            if (markers.x[mm] <= 300000 and markers.y[mm] > 7000 and markers.y[mm]<y_asth):
                markers.T[mm] = params.T_top + (T_asth - params.T_top)*(markers.y[mm] - 7000)/(y_asth - 7000)
            
	        # Transient left continent --> ocean geotherm
            if (markers.x[mm] > 300000 and markers.x[mm] < 350000 and markers.y[mm] > 7000 + (markers.x[mm]-300000)/50000*3000 and markers.y[mm] < y_asth):
		        # Continental geotherm
                T_cont = params.T_top + (T_asth-params.T_top)*(markers.y[mm]-(7000+(markers.x[mm]-300000)/50000*3000))/(y_asth-7000)
		        # T difference at the bottom of the oceanic and continental plates
                dt = -(params.T_top - T_asth)*(1-erf((y_asth-10000)/2/(kappa*age)**0.5))
		        # Oceanic geotherm
                T_ocea = T_asth+dt+(params.T_top-T_asth-dt)*(1-erf((markers.y[mm]-(7000+(markers.x[mm]-300000)/50000*3000))/2/(kappa*age)**0.5))
		        # Linear lateral transition
                mwt = (markers.x[mm]-300000)/50000
		        # Transitional temperate
                markers.T[mm] = T_cont*(1-mwt) + T_ocea*mwt

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
    P_first : ARRAY
        Array with 2 entries, specifying pressure BC.
    BC : BCs Class
        Object containing all boundary condition arrays for velocity, pressure and temperature 

    '''
    
    
    # instantiate a pre-populated parameters object
    params = Parameters()


    # set resolution of the grid
    xnum = 104+7
    ynum = 35+3


    # instantiate/load material properties object
    # file path must be relative to run.py file
    matData = np.loadtxt('./material_properties.txt', skiprows=3, delimiter=",")
    materials = Materials(matData)


    ###########################################################################    
    # Boundary conditions
    # instantiate the empty class
    BC = BCs(xnum, ynum)
    
    # pressure BCs - pressure in top-left node
    BC.P_first[1] = 1e5

    # velocity BCs
    BC.set_top_BC("free slip")
    BC.set_bottom_BC("free slip")
    BC.set_left_BC("free slip")
    BC.set_right_BC("free slip")


    # optional internal boundary
    BC.B_intern[0] = 102
    BC.B_intern[1] = 20
    BC.B_intern[2] = 23
    BC.B_intern[3] = (-5*1e-2)/(365.25*24*3600)  # convert to m/s 7.5
    BC.B_intern[4] = -1 
    BC.B_intern[5] = 0
    BC.B_intern[6] = 0
    BC.B_intern[7] = 0
    

    # temperature BCs
    BC.set_top_T_BC("fixed T", params.T_top)
    BC.set_bottom_T_BC("fixed T", params.T_bot)

    BC.set_left_T_BC("insulating")
    BC.set_right_T_BC("insulating")


    ###########################################################################
    # create grid object
    grid = Grid(xnum, ynum)

    # define grid points for (potentially) unevenly spaced grid
    updateGrid(params, grid, 0, params.tstp_max, BC.B_bottom)
    
    # TODO: check that if viscbox is in use the internal wall is inside it!
    

    ############################################################################
    # create markers
    mnumx = 550
    mnumy = 510
    markers = Markers(mnumx, mnumy)

    # initialize markers
    initialize_markers(markers, materials, params)
    
    return params, grid, materials, markers, BC
    



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
    
    # pull out the required parameters from params object
    bx = params.bx
    by = params.by
    Nx = params.Nx
    Ny = params.Ny
    
    non_uni_xsize = params.non_uni_xsize
    N_end = params.N_end
    b_end = params.b_end
    Ny_end = params.Ny_end
    by_end = params.by_end
    
    ###############################################################################
    # Horizontal grid
    
    # xsize - region of fixed grid at the end
    xsize_norm = params.xsize - N_end*b_end
    
    # grid point number at which the non-uniform grid ends
    xnum_ad = xnum - N_end

    if (t_curr==0):
        # set the points in the high res area
        # this only needs doing on the initial setup
        grid.x[Nx] = non_uni_xsize 
        for i in range(Nx+1,xnum_ad-Nx):  #xnum
            grid.x[i] = grid.x[i-1] + bx
            
        # size of the non-uniform region 
        D = xsize_norm - grid.x[xnum_ad-Nx-1]

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
            BC_bot[:,3] = 0
    
    # define factor of grid spacing to increase to the right of high res area
    # need to only do this for non-uniform def, otherwise div by 0!
    if (Nx > 0):
        F = 1.1
        # iteratively solve for F
        for i in range(0,200):
            F = (1 + D/bx*(1 - 1/F))**(1/Nx)
    
        # define grid points to the right of the high-res region
        for i in range(xnum_ad-Nx, xnum_ad):   #was xnum
            grid.x[i] = grid.x[i-1] + bx*F**(i-(xnum_ad-Nx-1))
            
        # we have a set of fixed resolution points at the upper edge of the grid 
        for i in range(xnum_ad, xnum):
            grid.x[i] = grid.x[i-1] + b_end
        
        if (t_curr==0):
            grid.x[xnum-1] = params.xsize
            grid.x[xnum_ad-1] = xsize_norm
    
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

    # ysize - region of fixed grid at the end
    ysize_norm = params.ysize - Ny_end*by_end

    # grid point number at which the non-uniform grid ends
    ynum_ad = ynum - Ny_end

    # set the high resolution area, assumes y[0] = 0
    for i in range(1,ynum_ad-Ny):
        grid.y[i] = grid.y[i-1] + by
      
    
    if (Ny > 0):
        # size of the non-uniform regions
        D = ysize_norm - grid.y[ynum_ad-Ny-1]
       
        # solve iteratively for scaling factor
        F = 1.1
        for i in range(0,100):
            F = (1 + D/by*(1 - 1/F))**(1/Ny)
            # set the grid points below the high-res region
        for i in range(ynum_ad-Ny, ynum_ad):
            grid.y[i] = grid.y[i-1] + by*F**(i-(ynum_ad-Ny-1))
        #we have a set of fixed resolution points between 200e3 and 400e3 m
        for i in range(ynum_ad, ynum):
            grid.y[i] = grid.y[i-1] + by_end

        # fix the end position if this is the first step
        if (t_curr==0):
            grid.y[ynum-1] = params.ysize
            grid.y[ynum_ad-1] = ysize_norm



spec_par = [
    ('gx', float64),
    ('gy', float64),
    ('Rgas', float64),
    ('xsize', float64),
    ('ysize', float64),
    ('T_min', float64),
    ('T_top', float64),
    ('T_bot', float64),
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
    ('bx', float64),
    ('by', float64),
    ('Nx', int64),
    ('Ny', int64),
    ('non_uni_xsize', float64),
    ('const', int64),
    ('N_end', int64),
    ('b_end', float64),
    ('Ny_end', int64),
    ('by_end', float64),
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
    T_top : FLOAT
        Temperature at the upper boundary.
    T_bot : FLOAT
        Temperature on the lower boundary.
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
    save_output : INT
        Number of steps between output, not currently implemented.
    save_fig : INT
        Number of steps between plotting of figures.
    output_name : STR
        The name of the folder to write the output/figures to.  This will be located
        in {output_path}/{output_name}.
    output_path : STR
        The output path where the result should be written to specified relative to the run.py file's location.
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
    N_end: INT
        Number of additional uniform grid points added at upper end of x-grid points
    b_end : FLOAT
        Spacing of additional uniform grid points added at upper end of the x-grid points
    viscbox : ViscBox Class object
        
    
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
        # physical constants
        self.gx = 0.0                           # x-direction gravitational acc
        self.gy = 9.81                          # y-direction gravitational acc
        self.Rgas = 8.314                       # gas constant
        
        # physical model setup
        self.xsize = 880000                     # physical size in x-direction
        self.ysize = 320000                     # physical size in y-direction
        
        self.T_min = 273                        # Minimum allowed temperature in the simulation
        self.T_top = self.T_min                 # temperature at the top face of the model (K)
        self.T_bot = 1825                       # temperature at the bottom face of the model (K)
        
        self.v_ext = 0.0                        # extension velocity of the grid (cm/yr)
        
        # viscosity model
        self.eta_min = 1e18                     # minimum viscosity
        self.eta_max = 1e25                     # maximum viscosity
        self.stress_min = 1e4                   # minimum stress
        self.eta_wt = 0                         # viscosity weighting, for (old?) visco-plastic model
        self.max_pow_law = 150                  # maximum power law exponent in visc model
        
        # timestepping
        self.t_end = 5e6*(365.24*24*3600)       # end time, simulation will exit if this is reached before max number of timesteps elapsed
        self.ntstp_max = 680                    # maximum number of timesteps 
        self.Temp_stp_max = 20                  # maximum number of temperature substeps
        
        self.tstp_max = 1e4*365.25*24*3600      # maximum timestep
        
        # marker options
        self.marker_max = 0.1                   # maximum marker movement per timestep (fraction of av. grid step)
        self.marker_sch = 1                     # marker scheme 0 = no movement, 1 = Euler, 4=RK4
        
        self.movemode = 0                       # velocity calculation 0 = momentum eqn, 1 = solid body (not working currently)
        
        # subgrid diffusion
        self.dsubgrid = 1                       # subgrid stress coeff (none if zero)
        self.dsubgridT = 1                      # subgrid diffusion coeff(none if zero)
        
        # switches for heating terms
        self.frict_yn = 1                       # use friction heating?
        self.adia_yn = 1                        # use adiabatic heating?
        
        
        # output options
        self.save_output = 50                    # number of steps between output files
        self.save_fig = 12                       # number of steps between figure output    
        self.output_name = "subductionBase"
        self.output_path = "../../Results/figures"
        
        #######################################################################
        # optional parameters, required based on model setup
        
        # grid spacing params - only required is using updateGrid()
        self.bx = 2200                          # x-grid spacing in high res area
        self.by = 2000                          # y-grid spacing in high res area
        self.Nx = 24                            # number of unevenly spaced grid pointss either side of high res zone
        self.Ny = 15                            # number of unvenly spaced grid points below high res zone
        self.non_uni_xsize = 175000             # physical x-size of non-uniform grid region left of the high res zone
        self.const = 1                          # flag which determines whether grid remains constant or not
        
        self.N_end = 7                          # number of additional uniform grid points at the upper edge of grid in x-direction
        self.b_end = 40e3                       # grid spacing in uniform region at upper edge
        self.Ny_end = 3                         # number of additional uniform grid points at the bottom of the grid in y-direction
        self.by_end = 40e3                      # grid spacing in uniform region at upper edge       

 
        # high viscosity box, if using, call constructor with 1 and set parameter values here
        self.viscbox = ViscBox(1)
        self.viscbox.xsize = 60000
        self.viscbox.ysize = 30000
        self.viscbox.xpos = 0.1666
    











    
