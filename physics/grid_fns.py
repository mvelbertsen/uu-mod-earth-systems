#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for calculating grid-based properties such as visco-elastic stresses, strain rates and grid spacings.
"""
import numpy as np
from numba import jit


@jit(nopython=True)
def viscElastStress(grid, grid0, tstep, xnum, ynum):
    '''
    Compute viscoelastic stresses for Stoke's solve.  New etas, stresses stored in grid0.

    Parameters
    ----------
    grid : Grid
        Grid object containing all the grid variables.
    grid0 : Grid
        Grid object containing all the previous step's grid variables, will be updated by this function.
    tstep : FLOAT
        Current timestep.
    xnum : INT
        x-resolution of the simulation domain.
    ynum : INT
        y-resolution of the simulation domain.

    Returns
    -------
    None.

    '''

    
    for i in range(0, ynum):
        for j in range(0, xnum):
            # viscoelastic factor
            xelvis = grid.eta_s[i,j]/(grid.eta_s[i,j] + tstep*grid.mu_s[i,j])
            grid0.eta_s[i,j] = grid.eta_s[i,j]*(1-xelvis)
            grid0.sigxy[i,j] = grid.sigxy[i,j]*xelvis
            
            # also do the normal comps
            if (i<ynum-1 and j<xnum-1):
                # viscoelastic factor
                xelvis = grid.eta_n[i,j]/(grid.eta_n[i,j] + tstep*grid.mu_n[i,j])
                grid0.eta_n[i,j] = grid.eta_n[i,j]*(1-xelvis)
                grid0.sigxx[i,j] = grid.sigxx[i,j]*xelvis


@jit(nopython=True)
def updateStresses(grid, tstep, xnum, ynum):
    '''
    Compute new viscoelastic stresses after Stoke's solve, with updated tstep.
    New stress values stored in sigxx2/xy2, also computes difference with old stresses.

    Parameters
    ----------
    grid : Grid
        Grid object containing all the grid variables, will be updated by this function..
    tstep : FLOAT
        Current timestep.
    xnum : INT
        x-resolution of the simulation domain.
    ynum : INT
        y-resolution of the simulation domain.

    Returns
    -------
    None.

    '''
    
    for i in range(0, ynum):
        for j in range(0, xnum):
            # viscoelastic factor
            xelvis = grid.eta_s[i,j]/(grid.eta_s[i,j] + tstep*grid.mu_s[i,j])
            grid.sigxy2[i,j] = grid.sigxy[i,j]*xelvis + 2*(1-xelvis)*grid.eta_s[i,j]*grid.epsxy[i,j]
            
            # also do the normal comps
            if (i<ynum-1 and j<xnum-1):
                # viscoelastic factor
                xelvis = grid.eta_n[i,j]/(grid.eta_n[i,j] + tstep*grid.mu_n[i,j])
                grid.sigxx2[i,j] = grid.sigxx[i,j]*xelvis + 2*(1-xelvis)*grid.eta_n[i,j]*grid.epsxx[i,j]
    
    # calculate differences
    grid.dsigxy = grid.sigxy2 - grid.sigxy
    grid.dsigxx = grid.sigxx2 - grid.sigxx




@jit(nopython=True)
def strainRateComps(grid, xnum, ynum):
    '''
    Calculates the strain rate components from the new grid velocities from the Stoke's solve.

    Parameters
    ----------
    grid : Grid
        Grid object containing all the grid variables, will be modified by this function.
    xnum : INT
        x-resolution of the simulation domain.
    ynum : INT
        y-resolution of the simulation domain.

    Returns
    -------
    None.

    '''
    
    for i in range(0,ynum):
        for j in range(0,xnum):
            
            
            if (i<ynum-1 and j<xnum-1):
                grid.epsxx[i,j] = 0.5*((grid.vx[i+1,j+1] - grid.vx[i+1,j])/grid.xstp[j] -\
                                       (grid.vy[i+1,j+1] - grid.vy[i,j+1])/grid.ystp[i])
            
            grid.epsxy[i,j] = 0.5*((grid.vx[i+1,j] - grid.vx[i,j])/grid.ystpc[i] +\
                                   (grid.vy[i,j+1] - grid.vy[i,j])/grid.xstpc[j])
            
            grid.espin[i,j] = 0.5*((grid.vy[i,j+1] - grid.vy[i,j])/grid.xstpc[j] -\
                                   (grid.vx[i+1,j] - grid.vx[i,j])/grid.ystpc[i])
                
            if (i>0 and j>0):
                grid.epsii[i-1,j-1] = np.sqrt(grid.epsxx[i-1,j-1]**2 + (grid.epsxy[i-1,j-1]**2\
                                    + grid.epsxy[i,j-1]**2 + grid.epsxy[i-1,j]**2\
                                    + grid.epsxy[i,j]**2)/4) 


def gridSpacings(bx, by, Nx, Ny, non_uni_xsize, xsize, ysize, grid, t_curr):
    '''
    Calculates the new grid point spacings based on the current xsize and ysize.
    This version contructs a non-uniform grid with a central-upper high resolution region
    and decreasing resolution outward from this.

    Parameters
    ----------
    bx : FLOAT
        Resolution of the high resolution zone in center of grid.
        If using uniform grid this should be the xsize/(xnum-1)
    by : FLOAT
        Resolution of the high resolution zone in center-top of grid.
        If using uniform grid this should be the ysize/(ynum-1)
    Nx : INT
        Size of non-uniform regions at edges on grid in x-direction.
    Ny : INT
        Size on non-uniform regions at edges of grid in y-direction.
    non_uni_xsize: FLOAT
        Size of the region covered by Nx points.
    xsize : FLOAT
        Physical size of the system in x-direction.
    ysize : FLOAT
        Physical size of the system in y-direction.
    grid : OBJ
        The grid object into which the new node positions will be written.
    t_curr : FLOAT
        The current simulation time, to determine whether to set up grid from scratch
        or extend an existing one.

    Returns
    -------
    None.

    '''
    
    xnum = grid.xnum
    ynum = grid.ynum
    
    ###############################################################################
    # Horizontal grid
    xsize_norm = 600e3

    if (t_curr==0):
        # set the points in the high res area
        # this only needs doing on the initial setup
        grid.x[Nx] = non_uni_xsize 
        for i in range(Nx+1,104-Nx):  #xnum
            grid.x[i] = grid.x[i-1] + bx
            
        # size of the non-uniform region 
        D = xsize_norm - grid.x[104-Nx-1] #xsize, xnum
        #print(grid.x[Nx+1])
        #print(grid.x[104-Nx-1])
        #print('D = ', D)

    else:
        # set the new position of the first node,
        # and the size of the non-uniform region
        grid.x[0] = grid.x[int(xnum/2)] - xsize/2
        D = xsize/2 - (grid.x[xnum-Nx-1] - grid.x[int(xnum/2)])
        
    
    # define factor of grid spacing to increase to the right of high res area
    # need to only do this for non-uniform def, otherwise div by 0!
    if (Nx > 0):
        F = 1.1
        # iteratively solve for F
        for i in range(0,200):
            F = (1 + D/bx*(1 - 1/F))**(1/Nx)
    
        # define grid points to the right of the high-res region
        for i in range(104-Nx, 104):   #was xnum
            grid.x[i] = grid.x[i-1] + bx*F**(i-(104-Nx-1))
        for i in range(104, xnum):
            grid.x[i] = grid.x[i-1] + 40e3
        
        if (t_curr==0):
            grid.x[xnum-1] = xsize
            grid.x[104-1] = xsize_norm
    
        # now do the same going leftward
        D = grid.x[Nx] - grid.x[0] # think this should still work for inital case?
    
        F = 1.1
        for i in range(0,100):
            F = (1 + D/bx*(1 - 1/F))**(1/Nx)
    
        # set the points left of the high res region
        for i in range(1,Nx):
            grid.x[i] = grid.x[i-1] + bx*F**(Nx+1-i)
        
    
    #print(grid.x)    
    ###########################################################################
    # Vertical grid

    # set the high resolution area
    for i in range(1,ynum-Ny):
        grid.y[i] = grid.y[i-1] + by
      
    
    if (Ny > 0):
        # size of the non-uniform regions
        D = ysize - grid.y[ynum-Ny-1]
       
        # solve iteratively for scaling factor
        F = 1.1
        for i in range(0,100):
            F = (1 + D/by*(1 - 1/F))**(1/Ny)
            # set the grid points below the high-res region
        for i in range(ynum-Ny, ynum):
            grid.y[i] = grid.y[i-1] + by*F**(i-(ynum-Ny-1))
           
        # fix the end position if this is the first step
        if (t_curr==0):
            grid.y[ynum-1] = ysize
