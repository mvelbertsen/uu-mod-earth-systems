#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definitions of data structures for thermomechanical model

"""

import numpy as np
from numba import jit, float64, int64
from numba.experimental import jitclass

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
    '''
    Class which stores all properties of markers in arrays.
    
    Attributes
    ----------
    xnum : INT
        Number of markers in x-direction, in initial configuration.
    ynum : INT
        Number of markers in y-direction, in initial configuration.
    num : INT
        Total number of markers (xnum*ynum)
    x : ARRAY
        Marker x-coordinates
    y : ARRAY
        Marker y-coordinates
    T : ARRAY
        Marker temperatures
    id : ARRAY
        Marker material type identifier
    nx : ARRAY
        Marker horizontal grid index of nearest top-left node
    ny : ARRAY
        Marker vertical grid index of the nearest top-left node
    sigmaxx : ARRAY
        Marker normal stress
    sigmaxy : ARRAY
        Marker shear stress
    eta : ARRAY
        Marker viscosity
    epsxx : ARRAY
        Marker normal strain rate
    epsxy : ARRAY
        Markers shear strain rate
    P : ARRAY
        Marker pressure
    gII : ARRAY
        Marker accumulated strain
    E_rat : ARRAY
        eii_marker/eii_grid ratio.
    epsii : ARRAY
        Marker deviatoric strain rate
    
    
    '''
    # creates empty marker property arrays
    def __init__(self, numx, numy):
        '''
        Constructor for the markers object, which creates zeroed arrays for all
        properties, ready to be intialized by an implementation of initialize_markers

        Parameters
        ----------
        numx : INT
            Number of markers in x-direction in initial grid configuration.
        numy : INT
            Number of markers in y-direction in initial grid configuration.

        Returns
        -------
        None.

        '''

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
    '''
    Class which stores the contents of material_properties.txt file

    Attributes
    ----------
    num_materials : INT
        number of materials listed in material_properties.txt
    rho : ARRAY
        Density parameters for each material: density, thermal expansion, compressibility
    visc : ARRAY
        Viscosity parameters fro each material:
        model choice, cons visc, Ad, n, Ea, Va
    mu : ARRAY
        Shear modulus for each material.
    plast : ARRAY
        Plasticity parameters for each material:
        C0, C1, sin(FI0), sin(FI1), gamma0, gamma1
    Cp : ARRAY
        Specific heat capacity for each material.
    kT : ARRAY
        Thermal conductivity parameters for each material:
        k0, a
    radH : ARRAY
        Radiogenic heat production for each material.
    
    '''
    
    # creates and fills material properties from specified file
    def __init__(self, materialData):
        '''
        Constructor for the Materials class.

        Parameters
        ----------
        materialData : ARRAY
            Contents of the material_properties.txt file, loaded into a np array
            using np.loadtxt.

        Returns
        -------
        None.

        '''
        
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
        # power law visc, eta = Ad * sigma**n * exp(-(Ea + Va*P)/RT)
        # 2 - Ad
        # 3 - n
        # 4 - Ea
        # 5 - Va
        self.visc = materialData[:,3:9]
        
        # shear modulus
        self.mu = materialData[:,9] 
        
        # plasticity parameters
        # for details see eqns 14.10-12 in Gerya
        
        # 0 - C0 - cohesion, pre strain weakening
        # 1 - C1 - cohesion, post strain weakening
        # 2 - sin(FI0) - internal friction, pre-stain weakening 
        # 3 - sin(FI1) - internal friction, post-strain weakening
        # 4 - Gamma0 - for sig_ii < Gamma0, cohesion and friction = C0, sin(F0)
        # 	       for gamma0 < sigii < gamma1, coh, frict given by eqn 14.10
        # 5 - Gamma1 - for sigii > Gamma1, cohesion and friction = C0, sin(F0)
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
    '''
    Class which contains all the grid quantities.
    
    Attributes
    ----------
    xnum : INT
        Number of nodes in x-direction.
    ynum : INT
        Number of nodes in y-direction.
    x : ARRAY
        x-diection positions of basic nodes
    y : ARRAY
        y-direction positions of basic nodes.
    xstp : ARRAY
        x-direction spacing between basic nodes.
    ystp : ARRAY
        y-direction spacing between basic nodes.
    rho : ARRAY
        Density.
    rhoCp : ARRAY
        density * heat capacity, for use in Temperature solver
    T : ARRAY
        Temperature
    kT : ARRAY
        Thermal conductivity
    H_a : ARRAY
        Adiabatic heat
    H_r : ARRAY
        Radiogenic heat
    wt : ARRAY
        weights for markers to grid interpolation of basic properties
    eta_s : ARRAY
        Viscosity for shear stress calculation.
    wt_eta_s : ARRAY
        weights for markers to interpolation of shear stress quantities
    sigxy : ARRAY
        Shear stresses
    mu_s : ARRAY
        shear modulus for shear stresses
    epsxy : ARRAY
        shear strain rate
    sigxy2 : ARRAY
        Updated shear stress (after Stoke's solve).
    dsigxy : ARRAY
        shear stress difference, for subgrid stress difference.
    espin : ARRAY
        spin tensor (for rotation of stress components).
    P : ARRAY
        Pressure
    eta_n : ARRAY
        Viscosity for normal stress calculations.
    wt_eta_n : ARRAY
        Weights for markers to interpolation of normal stress quantities
    sigxx : ARRAY
        Normal stresses
    mu_n : ARRAY
        shear modulus for normal stresses
    epsxx : ARRAY
        Normal strain rate
    sigxx2 : ARRAY
        Updated normal stress (after Stoke's solve).
    dsigxx : ARRAY
        Normal stress difference, for subgrid stress difference.
    espii : ARRAY
        Deviatoric strain rate.
    vx : ARRAY
        x-direction velocities
    vy : ARRAY
        y-direction velocities
    cx : ARRAY
        cell-centered node x-positions
    cy : ARRAY
        cell-centered node y-positions
    xstpc : ARRAY
        centered node x spacings
    ystpc : ARRAY
        centered node y spacings
        
    
    '''
    
    
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
    
