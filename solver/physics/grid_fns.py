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


@jit(nopython=True)
def basicGridVelocities(gridvx, gridvy, xnum, ynum):
    '''
    Interpolates the velocity values to the basic nodes, for visualisation only.

    Parameters
    ----------
    gridvx : ARRAY
        x-velocities at the staggered computation nodes.
    gridvy : ARRAY
        y-velocities at the staggered computation nodes.
    xnum : INT
        x-resolution of the simulation domain.
    ynum : INT
        y-resolution of the simulation domain.

    Returns
    -------
    vxb : ARRAY
        x-velocities at the basic nodes.
    vyb : ARRAY
        y-velocities at the basic nodes.

    '''
    vxb = np.zeros((ynum, xnum))
    vyb = np.zeros((ynum, xnum))
    
    for i in range(0,ynum):
        for j in range(0,xnum):
            vxb[i,j] = (gridvx[i,j] + gridvx[i+1,j])/2
            vyb[i,j] = (gridvy[i,j] + gridvy[i,j+1])/2
    
    return vxb, vyb