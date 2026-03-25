#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

choice of model is made by importing an initializeModel implementation
from the chosen model directory

"""

import numpy as np
from copy import deepcopy


# load the component fucntions from their respective files
from solver.dataStructures import copyGrid

from solver.physics.StokesContinuitySolver import StokesContinuitySolver, constructStokesRHS
from solver.physics.TemperatureSolver import TemperatureSolver, constructTempRHS
from solver.physics.markers_fns import markersToGrid, gridToMarker, updateMarkerErat, subgridStressChanges,\
                        subgridDiffusion, advectMarkers
from solver.physics.grid_fns import updateStresses, viscElastStress, strainRateComps




def step(params, grid, materials, markers, P_first, B_top, B_bottom,\
             B_left, B_right, B_intern, BT_top, BT_bottom, BT_left, BT_right, timestep, ntstp, grid0, debug):
    """
    Performs a timestep of the solver, not including visualisation and grid changes which are model-dependent.
    
    Should be used in a timestep loop along with custom plotting and grid updates (if required).  
    
    See models/simpleStokes/run.py for example usage.
    

    Parameters
    ----------
    params : jit class
        Class object containing the required model parameters.
    grid : Grid object
        Initialised grid object.
    materials : Materials object
        Initialised materials object.
    markers : Markers object
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
    timestep : FLOAT
        The current simulation timestep (this is reset and recalculated every step)
    ntstp : INT
        Current timestep number.
    grid0 : grid Object
        Initialised grid object for storing old timestep's values
    debug : INT
        Flag to say if debug statements should be printed
    

    Returns
    -------
    None.

    """
    
    # for ease of use, set xnum, ynum
    xnum = grid.xnum
    ynum = grid.ynum
    
    
    
    # compute average step size
    xstp_av = params.xsize/(xnum-1)
    ystp_av = params.ysize/(ynum-1)
    
    

    # store old grid values
    if (debug):
        print('copying old array')
    copyGrid(grid, grid0)
    
    # and clear all arrays
    grid.rho = np.zeros((ynum,xnum))
    grid.T = np.zeros((ynum,xnum))
    grid.kT = np.zeros((ynum,xnum))
    grid.H_a = np.zeros((ynum,xnum))
    grid.H_r = np.zeros((ynum,xnum))
    grid.rhoCP = np.zeros((ynum,xnum))
    grid.wt = np.zeros((ynum,xnum))
    
    grid.eta_s = np.zeros((ynum,xnum))
    grid.mu_s = np.zeros((ynum,xnum))
    grid.sigxy = np.zeros((ynum,xnum))
    grid.wt_eta_s = np.zeros((ynum,xnum))
    
    grid.eta_n = np.zeros((ynum-1,xnum-1))
    grid.mu_n = np.zeros((ynum-1,xnum-1))
    grid.sigxx = np.zeros((ynum-1,xnum-1))
    grid.wt_eta_n = np.zeros((ynum-1,xnum-1))
    
    # reset plastic yielding flag
    plast_y = 0
    
    # set timestep to max, will be reduced if max velocities require it
    timestep = params.tstp_max
    
    # compute the grid spacings for all nodes
    # grid steps for the basic nodes
    grid.xstp = grid.x[1:] - grid.x[:-1]
    grid.ystp = grid.y[1:] - grid.y[:-1]
    
    # grid points for the center nodes
    # horizontal
    grid.cx[0] = grid.x[0] - grid.xstp[0]/2
    grid.cx[1:xnum] = (grid.x[1:] + grid.x[:-1])/2
    grid.cx[xnum] = grid.x[xnum-1] + grid.xstp[xnum-2]/2
    # vertical
    grid.cy[0] = grid.y[0] - grid.ystp[0]/2
    grid.cy[1:ynum] = (grid.y[1:] + grid.y[:-1])/2
    grid.cy[ynum] = grid.y[ynum-1] + grid.ystp[ynum-2]/2
    
    # grid spacing for center nodes
    grid.xstpc[0] = grid.xstp[0]
    grid.ystpc[0] = grid.ystp[0]
    
    grid.xstpc[xnum-1] = grid.xstp[xnum-2]
    grid.ystpc[ynum-1] = grid.ystp[ynum-2]
    
    grid.xstpc[1:xnum-1] = (grid.x[2:] - grid.x[:xnum-2])/2
    grid.ystpc[1:ynum-1] = (grid.y[2:] - grid.y[:ynum-2])/2
    
    if (debug):
        print('marker to grid interp.')
    # interpolate parameters from markers to nodes + compute viscosities
    markersToGrid(markers, materials, grid, grid0, xnum, ynum, params, timestep, ntstp, plast_y, B_intern)
    
    
    # apply thermal BCs for interpolated T
    # upper boundary
    grid.T[0,1:xnum-1] = BT_top[1:xnum-1,0] + BT_top[1:xnum-1,1]*grid.T[1,1:xnum-1]
    # lower boundary
    grid.T[ynum-1,1:xnum-1] = BT_bottom[1:xnum-1,0] + BT_bottom[1:xnum-1,1]*grid.T[ynum-2,1:xnum-1]
    # left boundary
    grid.T[:,0] = BT_left[:,0] + BT_left[:,1]*grid.T[:,1]
    # right boundary
    grid.T[:,xnum-1] = BT_right[:,0] + BT_right[:,1]*grid.T[:,xnum-2]
    
    # then interpolate back to markers - only if it is t=0!
    if (ntstp==0):
        gridToMarker([grid.T], [markers.T], markers.x, markers.y, markers.nx, markers.ny, grid)
    
    # compute viscoelastic visc and stress    
    viscElastStress(grid, grid0, timestep, xnum, ynum)
                
    # Compute RHS of Stokes+cont
    R_x, R_y, R_C = constructStokesRHS(grid, grid0, params, xnum, ynum)

    ###########################################################################
    # compute velocity, pressure fields
    if (params.movemode==0):
        # call Stoke's solver for v, P
        if (debug):
            print('entering Stokes')
        grid.vx, grid.vy, grid.P, resvx, resvy, resP = StokesContinuitySolver(P_first, grid0.eta_s, grid0.eta_n, xnum, ynum, grid.x, grid.y, R_x, R_y, R_C, B_top, B_bottom, B_left, B_right, B_intern)
    else:
        raise ValueError("Unrecognised movemode value, accepted values are 0 (Stokes eqns). 1=solid body rot not implemented in this version!")
    
    # compute strain rate tensor comps
    strainRateComps(grid, xnum, ynum)
    
    # check velocity maxima
    vxmax = max(abs(np.max(grid.vx)), abs(np.min(grid.vx)))
    vymax = max(abs(np.max(grid.vy)), abs(np.min(grid.vy)))
    
    # update timestep based on maximal velocities
    if (vxmax > 0):
        if (timestep > params.marker_max*xstp_av/vxmax):
            timestep = params.marker_max*xstp_av/vxmax
    else:
        raise ValueError("Negative vxmax!")
    
    if (vymax > 0):
        if (timestep > params.marker_max*ystp_av/vymax):
            timestep = params.marker_max*ystp_av/vymax
    else:
        raise ValueError("Negative vymax!")
    
    if (debug):
        print('stresses')
    # compute new stresses using the new displacement timestep
    updateStresses(grid, timestep, xnum, ynum)
       
    # compute strain rates+pressure for markers
    gridToMarker([grid.epsxy], [markers.epsxy],\
                 markers.x, markers.y, markers.nx, markers.ny, grid)
    
    gridToMarker([grid.epsxx, grid.P], [markers.epsxx, markers.P],\
                 markers.x, markers.y, markers.nx, markers.ny, grid, node_type=1)
    
    # then the epsii/rat, which has it's own correction
    updateMarkerErat(markers, materials, grid, timestep)
    
    # compute subgrid stress changes for markers
    if (params.dsubgrid>0):
        subgridStressChanges(markers, grid, xnum, ynum, materials, params, timestep)
        
    # then update stress changes
    dsxym = np.zeros((markers.num))
    dsxxm = np.zeros((markers.num))
    
    gridToMarker([grid.dsigxy], [dsxym], markers.x, markers.y, markers.nx, markers.ny, grid)
    markers.sigmaxy += dsxym
    
    gridToMarker([grid.dsigxx], [dsxxm], markers.x, markers.y, markers.nx, markers.ny, grid, node_type=1)
    markers.sigmaxx += dsxxm
    
    ###########################################################################
    # Temperature eqn
    if (timestep>0 and params.Temp_stp_max>0):
        if (debug):
            print('temperature solve')
        # compute RHS for the temperature eqn from heating terms
        RT = constructTempRHS(grid, params)
        
        # set temperature timestep
        timestep_T = timestep
        
        # set total temperature timesteps
        steps_T = 0
        
        # set old temp
        T0 = deepcopy(grid.T)
        # loop for sub timestepping for temperature eqn
        while(steps_T < timestep):
            
            # solve the temperature eqn
            T2, T2res = TemperatureSolver(timestep_T, xnum, ynum, grid.x, grid.y, grid.kT, grid.rhoCP,\
                                                BT_top, BT_bottom, BT_left, BT_right, RT, T0)
            
            # compute temp changes
            dT = T2 - T0
            
            # check whether it is above max temp changes
            dTmax = np.max(np.abs(dT))
            
            # if we are over the maximum allowed T change, reduce timestep
            if (dTmax > params.Temp_stp_max):
                timestep_T = timestep_T*params.Temp_stp_max/dTmax
                
                # solve again for T
                T2, T2res = TemperatureSolver(timestep_T, xnum, ynum, grid.x, grid.y, grid.kT, grid.rhoCP,\
                                                    BT_top, BT_bottom, BT_left, BT_right, RT, T0)
            
            # add to the total timestep length
            steps_T += timestep_T
            
            # compute next timestep
            if (timestep_T > timestep-steps_T):
                # shorten so that we don't overshoot
                timestep_T = timestep - steps_T
            
            # update old T 
            T0 = deepcopy(T2)
       
        # compute temp changes
        dTn = T2 - grid.T
        
        if (params.dsubgridT>0):
            # compute subgrid diffusion for markers
            dTsg = subgridDiffusion(grid, markers, params, xnum, ynum, timestep, xstp_av, ystp_av)
        
            # the corrected T diff for the nodal points 
            dTn = dTn - dTsg
    
        # update T for markers
        dTm = np.zeros((markers.num))
        gridToMarker([dTn], [dTm], markers.x, markers.y, markers.nx, markers.ny, grid)
        markers.T += dTm
    
    if (debug):
        print('advection')
    ###########################################################################
    # move markers using vel field
    advectMarkers(markers, grid, xnum, ynum, timestep, params.marker_sch)
    