#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definitions of data structures for thermomechanical model

"""

import numpy as np
from numba import jit, float64, int64
from numba.experimental import jitclass

spec_par = [
    ('gx', float64),
    ('gy', float64),
    ('Rgas', float64),
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
    ('bx', float64),
    ('by', float64),
    ('Nx', int64),
    ('Ny', int64),
    ('non_uni_xsize', float64),
    ('save_output', int64),
    ('save_fig', int64),
]
@jitclass(spec_par)
class Parameters():
    
    # creates parameters object
    def __init__(self):
        
        # physical constants
        self.gx = 0.0                           # x-direction gravitational acc
        self.gy = 9.81                          # y-direction gravitational acc
        self.Rgas = 8.314                       # gas constant
        
        # physical model setup
        self.T_top = 273                        # temperature at the top face of the model (K)
        self.T_bot = 1750                       # temperature at the bottom face of the model (K)
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


spec_mark = [
     ('xnum', int64),
     ('ynum', int64),
     ('num', int64),
     ('x', float64[:]),
     ('y', float64[:]),
     ('T', float64[:]),
     ('id', int64[:]),
     ('nx', int64[:]),
     ('ny', int64[:]),
     ('sigmaxx', float64[:]),
     ('sigmaxy', float64[:]),
     ('eta', float64[:]),
     ('epsxx', float64[:]),
     ('epsxy', float64[:]),
     ('P', float64[:]),
     ('gII', float64[:]),
     ('E_rat', float64[:]),
     ('epsii', float64[:]),
]

@jitclass(spec_mark)
class Markers():
    
    # creates empty marker property arrays
    def __init__(self, numx, numy):
        
        # number of markers in each direction in initial setup
        self.xnum = numx
        self.ynum = numy
        
        # total number of markers
        self.num = self.xnum*self.ynum

        # arrays storing marker properties
        self.x = np.zeros((self.num))               # x-coordinate
        self.y = np.zeros((self.num))               # y-coordinate
        self.T = np.zeros((self.num))               # Temperature
        self.id = np.zeros(self.num, dtype=np.int64)    # material type
        self.nx = np.zeros(self.num, dtype=np.int64)    # horizontal grid index
        self.ny = np.zeros(self.num, dtype=np.int64)    # vertical grid index
        self.sigmaxx = np.zeros((self.num))         # normal stress
        self.sigmaxy = np.zeros((self.num))         # shear stress
        self.eta = np.zeros((self.num))             # viscosity
        self.epsxx = np.zeros((self.num))           # normal strain rate
        self.epsxy = np.zeros((self.num))           # shear strain rate
        self.P = np.zeros((self.num))               # pressure
        self.gII = np.zeros((self.num))             # accumulated strain
        self.E_rat = np.zeros((self.num))           # eiimarker/eiigrid ratio 
        self.epsii = np.zeros((self.num))           # deviatoric strain rate

        
spec_mat = [
    ('num_materials', int64),
    ('rho', float64[:,:]),
    ('visc', float64[:,:]),
    ('mu', float64[:]),
    ('plast', float64[:,:]),
    ('Cp', float64[:]),
    ('kT', float64[:,:]),
    ('radH', float64[:]),
]
@jitclass(spec_mat)       
class Materials():
    
    # creates and fills material properties from specified file
    def __init__(self, materialData):
        
        # load the material parameters from a file
        #materialData = np.loadtxt(filename, skiprows=3, delimiter=",")
        
        self.num_materials = materialData.shape[0]
        
        # load the material parameters into arrays - as in original code
        # density related properties
        # 0 - standard density
        # 1 - thermal expansion
        # 2 - compressibility
        self.rho = materialData[:,0:3]
        
        # viscosity model parameters
        # 0 - choice of viscosity model (0 = constant visc, 1 = power law)
        # 1 - viscosity for constant model
        # 2 - Ad
        # 3 - n
        # 4 - Ea
        # 5 - Va
        self.visc = materialData[:,3:9]
        
        # shear modulus
        self.mu = materialData[:,9] 
        
        # plasticity parameters
        # 0 - C0
        # 1 - C1
        # 2 - sin(FI0)
        # 3 - sin(FI1)
        # 4 - Gamma0
        # 5 - Gamma1
        self.plast = materialData[:,10:16]
        
        # Specific heat capacity
        self.Cp = materialData[:,16] 
        
        # thermal conductivity parameters
        # 0 - k0
        # 1 - a
        self.kT = materialData[:, 17:19] 
        
        # radiogenic heat production (W/m**3)
        self.radH = materialData[:,19] 
        

spec_grid = [
    ('xnum', int64),
    ('ynum', int64),
    ('x', float64[:]),
    ('y', float64[:]),
    ('xstp', float64[:]),
    ('ystp', float64[:]),
    ('rho', float64[:,:]),
    ('rhoCP', float64[:,:]),
    ('T', float64[:,:]),
    ('kT', float64[:,:]),
    ('H_a', float64[:,:]),
    ('H_r', float64[:,:]),
    ('wt', float64[:,:]),
    ('eta_s', float64[:,:]),
    ('wt_eta_s', float64[:,:]),
    ('sigxy', float64[:,:]),
    ('mu_s', float64[:,:]),
    ('epsxy', float64[:,:]),
    ('sigxy2', float64[:,:]),
    ('dsigxy', float64[:,:]),
    ('espin', float64[:,:]),
    ('P', float64[:,:]),
    ('eta_n', float64[:,:]),
    ('wt_eta_n', float64[:,:]),
    ('sigxx', float64[:,:]),
    ('mu_n', float64[:,:]),
    ('epsxx', float64[:,:]),
    ('sigxx2', float64[:,:]),
    ('dsigxx', float64[:,:]),
    ('epsii', float64[:,:]),
    ('vx', float64[:,:]),
    ('vy', float64[:,:]),
    ('cx', float64[:]),
    ('cy', float64[:]),
    ('xstpc', float64[:]),
    ('ystpc', float64[:]),
]

@jitclass(spec_grid)
class Grid():
    
    # creates empty grid structures
    def __init__(self, xnum, ynum):
        
        self.xnum = xnum
        self.ynum = ynum
        
        #######################################################################
        # basic node coordinates
        self.x = np.zeros((xnum))
        self.y = np.zeros((ynum))
        
        # basic node steps
        self.xstp = np.zeros((xnum-1))
        self.ystp = np.zeros((ynum-1))
        
        # arrays for properties stored at the basic nodes
        self.rho = np.zeros((ynum, xnum))           # density
        self.rhoCP = np.zeros((ynum, xnum))         # density * C_p
        
        self.T = np.zeros((ynum, xnum))             # temperature (K)
        self.kT = np.zeros((ynum, xnum))            # thermal conductivity
        self.H_a = np.zeros((ynum, xnum))           # adiabatic heating
        self.H_r = np.zeros((ynum, xnum))           # radiogenic heating
        
        self.wt = np.zeros((ynum, xnum))            # weights for marker interpolation
        
        self.eta_s = np.zeros((ynum, xnum))         # viscosity for shear stress
        self.wt_eta_s = np.zeros((ynum, xnum))      # weights for interpolation
        
        self.sigxy = np.zeros((ynum, xnum))         # shear stress
        self.mu_s = np.zeros((ynum, xnum))          # shear modulus for shear stress
        self.epsxy = np.zeros((ynum, xnum))         # shear strain rate
        
        self.sigxy2 = np.zeros((ynum, xnum))        # udpated shear stress (after Stoke's solve)
        self.dsigxy = np.zeros((ynum, xnum))        # shear stress difference (for subgrid stress calculations)
        
        self.espin = np.zeros((ynum, xnum))         # spin tensor (for rotation fo stress components)
        
        #######################################################################
        # pressure nodes
        self.P = np.zeros((ynum-1, xnum-1))         # pressure 
        
        self.eta_n = np.zeros((ynum-1, xnum-1))     # viscosity for normal stresses
        self.wt_eta_n = np.zeros((ynum-1, xnum-1))  # weights for marker interpolation
        
        self.sigxx = np.zeros((ynum-1, xnum-1))     # normal stress
        self.mu_n = np.zeros((ynum-1, xnum-1))      # shear modulus for normal stress
        self.epsxx = np.zeros((ynum-1, xnum-1))     # normal strain rate
        
        self.sigxx2 = np.zeros((ynum-1, xnum-1))    # updated normal stress (after Stoke's solve)
        self.dsigxx = np.zeros((ynum-1, xnum-1))    # normal stress difference (for subgrid stress calculations)
        self.epsii = np.zeros((ynum, xnum))         # deviatoric strain rate
        
        #######################################################################
        # velocities
        self.vx = np.zeros((ynum+1, xnum))          # x-velocity
        self.vy = np.zeros((ynum, xnum+1))          # y-velocity
        
        # grid positions for centers
        self.cx = np.zeros((xnum+1))                # centered node x-position
        self.cy = np.zeros((ynum+1))                # centered node y-position
        
        # grid spacings
        self.xstpc = np.zeros((xnum))               # centered node x-spacing
        self.ystpc = np.zeros((ynum))               # centered node y-spacing
        

@jit
def copyGrid(grid, grid0):
    '''
    manual copy function for the Grid object, copies grid into grid0.  For compatibility with jit.

    Parameters
    ----------
    grid : Grid
        The new grid object to be copied from.
    grid0 : Grid
        The previous timestep grid object to be copied to.

    Returns
    -------
    None.

    '''
    
    grid0.x = grid.x.copy()
    grid0.y = grid.y.copy()
    
    grid0.xstp = grid.xstp.copy()
    grid0.ystp = grid.ystp.copy()
    
    grid0.xstpc = grid.xstpc.copy()
    grid0.ystpc = grid.ystpc.copy()
    
    grid0.rho = grid.rho.copy()
    grid0.rhoCP = grid.rhoCP.copy()
    grid0.T = grid.T.copy()
    grid0.kT = grid.kT.copy()
    grid0.H_a = grid.H_a.copy()
    grid0.H_r = grid.H_r.copy()
    grid0.wt = grid.wt.copy()
    
    grid0.eta_s = grid.eta_s.copy()
    grid0.wt_eta_s = grid.wt_eta_s.copy()
    grid0.sigxy = grid.sigxy.copy()
    grid0.mu_s = grid.mu_s.copy()
    grid0.epsxy = grid.epsxy.copy()
    grid0.sigxy2 = grid.sigxy2.copy()
    grid0.dsigxy = grid.dsigxy.copy()
    grid0.espin = grid.espin.copy()
    
    grid0.P = grid.P.copy()
    grid0.eta_n = grid.eta_n.copy()
    grid0.wt_eta_n = grid.wt_eta_n.copy()
    grid0.sigxx = grid.sigxx.copy()
    grid0.mu_n = grid.mu_n.copy()
    grid0.epsxx = grid.epsxx.copy()
    grid0.sigxx2 = grid.sigxx2.copy()
    grid0.dsigxx = grid.dsigxx.copy()
    grid0.epsii = grid.epsii.copy()

    grid0.vx = grid.vx.copy()
    grid0.vy = grid.vy.copy()
    
    grid0.cx = grid.cx.copy()
    grid0.cy = grid.cy.copy()
    
