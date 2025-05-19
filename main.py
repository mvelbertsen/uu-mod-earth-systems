#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main file for the Visco-elastic-plastic model - run this to run the model!

choice of model is made by importing an initializeModel implementation
from the chosen model directory

"""

import numpy as np
from numba import jit
from copy import deepcopy
from time import time
import os

# if debugging, this should be 1 AND jitclass tags in dataStructures must be commented out!
os.environ["NUMBA_DISABLE_JIT"] = "0"
# if 1 prints out extra statements at various places in the timeloop
debug = 0

# load the component fucntions from their respective files
from dataStructures import Markers, Materials, Grid, Parameters, copyGrid

from physics.StokesContinuitySolver import StokesContinuitySolver, constructStokesRHS
from physics.TemperatureSolver import TemperatureSolver, constructTempRHS
from physics.markers_fns import markersToGrid, gridToMarker, updateMarkerErat, subgridStressChanges,\
                        subgridDiffusion, advectMarkers
from physics.grid_fns import updateStresses, viscElastStress, strainRateComps, gridSpacings

from visualisation import plotAVar, plotSeveralVars, plotMarkerFields, basicGridVelocities

# load the setup fn for the chosen model
from models.lithosphereExtension.setup import initializeModel




strt = time()
###############################################################################
# initialize the model setup 
params, grid, materials, markers, xsize, ysize, P_first, B_top, B_bottom,\
B_left, B_right, B_intern, BT_top, BT_bottom, BT_left, BT_right = initializeModel()


# for ease of use, set xnum, ynum
xnum = grid.xnum
ynum = grid.ynum

# initialize grid0 for old values
grid0 = Grid(xnum, ynum)

# compute average step size
xstp_av = xsize/(xnum-1)
ystp_av = ysize/(ynum-1)

# initialize timesteping
time_curr = 0
timestep = params.tstp_max


###############################################################################
# begin time loop
for nt in range(0, params.ntstp_max):
    
    # store old grid values
    if (debug):
        print('copying old array')
    copyGrid(grid, grid0)
    
    #grid0 = deepcopy(grid)
    
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
    markersToGrid(markers, materials, grid, grid0, xnum, ynum, params, timestep, nt, plast_y)
    
    
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
    if (time_curr==0):
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
    
    ###########################################################################
    # visualization + output    
    if (nt%(params.save_fig)==0):
        print('plotting')
        
        # interpolate vx, vy for basic grid
        vxb, vyb = basicGridVelocities(grid.vx, grid.vy, xnum, ynum)
        
        # plotting
        plotAVar(grid, vxb, vyb, xsize, ysize, nt, time_curr)
        plotSeveralVars(grid, vxb, vyb, xsize, ysize, nt, time_curr)
        plotMarkerFields(xsize, ysize, markers, grid, nt, time_curr)
    
    
    ###########################################################################
    # advance timestep
    time_curr += timestep
    print('Time: %.3f Myr'%(time_curr*1e-6/(365.25*24*3600)))
    
    if (debug):
        print('updating grid spacings')
    
    # update grid positions based on extension
    ysize += -params.v_ext/xsize*ysize*timestep
    xsize += params.v_ext*timestep
    
    gridSpacings(params.bx, params.by, params.Nx, params.Ny, params.non_uni_xsize, xsize, ysize, grid, time_curr)
    
    # if we have changing grid, need to update bottom BC
    if (abs(params.v_ext)>0):
        B_bottom[:,2] = -params.v_ext/xsize*ysize
        B_bottom[:,3] = 0

end = time() - strt
print('time elapsed: %f'%(end))    

