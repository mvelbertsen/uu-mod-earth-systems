#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File which contains the setup for a lithospheric extension model

"""

import numpy as np
import pathlib

from dataStructures import Markers, Parameters, Grid, Materials
from physics.grid_fns import gridSpacings

import matplotlib.pyplot as plt


def mountain_slope_curve(x, xsize, ysize, params):
    '''
    Define bedrock slope as concave up decreasing curve. Follows roughly bedrock slope of 10% (y=0.1x).

    '''

    # a = (xsize/ysize*0.1) * ysize/(xsize**2)   # coefficient
    # b = ysize/5*4                                # vertical offset
    # mountain_slope = -a * x**2 + b      # note horizontal shift along x-axis (of -xsize)

    # return mountain_slope
    # return params.by+0.1*x
    # return params.by+0.2*x
    # return ysize/3 + 0.15*x
    # return params.bx + 0.15*x                                                                                  # straight line
    # return (ysize-params.bx) - (ysize-params.bx*2)/(xsize**2)*(x-xsize)**2                                     # quadratic - steeper
    return ((ysize-params.bx) - (ysize-params.bx*2)/(xsize**2)*(x-xsize)**2)*0.5 + (1-0.5)*(params.bx + 0.15*x)  # quadratic - less steep


def glacier_surface_curve(x, xsize, ysize, params):
    '''
    Define glacier surface slope as concave down decreasing curve.

    '''

    # a =  -(xsize/ysize*0.1) * ysize/((xsize/10*8)**2)   # coefficient
    # b = 1-(xsize/10*0.8) + ysize/5*4                         # vertical offset
    # # glacier_surface = -a * (x-xsize)**2 + b
    # # glacier_surface = -a * (x+xsize/4)**2 #+ 575#b-200

    # # glacier_surface = params.by + ysize/(xsize**2) * x**2 #+ 575#b-200
    # glacier_surface = params.by + 0.2*(ysize**3)/(xsize**5) * x**4 #+ 575#b-200

    # return glacier_surface
    # # return ysize/2 + 1-(ysize/8*7) + ysize/xsize**2 * x**2
    # return ysize/(xsize**4)*x**4 + ysize/3
    return 5*ysize/(xsize**4)*x**4 + params.bx



def initialize_markers(markers, materials, params, xsize, ysize):
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
    xsize : FLOAT
        Physical x-size of the simulation domain.
    ysize : FLOAT
        Physical y-size of the simulation domain.

    Returns
    -------
    None.

    '''
    
    np.random.seed(1337)
    # set the rough grid spacing for the markers
    mxstp = xsize/markers.xnum
    mystp = ysize/markers.ynum
    
    # marker number index
    mm = 0
    
    for j in range(0,markers.xnum):
        for i in range(0,markers.ynum):
            
            # set coordinates as grid + small random displacement
            markers.x[mm] = (j + np.random.random())*mxstp
            markers.y[mm] = (i + np.random.random())*mystp
                       

            # # bedrock
            # if markers.y[mm] >= topography_curve(markers.x[mm], xsize, ysize):
            #     markers.id[mm] = 2
            #     markers.T[mm] = 273

            # # glacier
            # elif markers.y[mm] >= ice_curve(markers.x[mm], xsize, ysize):
            #     markers.id[mm] = 1
            #     markers.T[mm] = 263

            # # air         
            # else: 
            #     dtdy = 0.65/1000 # approximate adiabetic lapse rate for the air K/m
            #     markers.id[mm] = 0 
            #     markers.T[mm] = 253 - dtdy*(ysize - markers.y[mm])


            # # now define type based on location, to give our initial setup
            # # sticky air
            # markers.id[mm] = 0
            # markers.T[mm]  = 273        # 0 °C
            
            # bedrock
            if markers.y[mm] >= mountain_slope_curve(markers.x[mm], xsize, ysize, params) or markers.y[mm] >= ysize-params.by:
                markers.id[mm] = 1
                markers.T[mm]  = 273    # 0 °C
            
            # glacier   # try to reproduce glacier shape from SIA
            elif markers.y[mm] >= glacier_surface_curve(markers.x[mm], xsize, ysize, params):
                markers.id[mm] = 2
                markers.T[mm]  = 263    # -10 °C

            # air         
            else: 
                dtdy = 6.5/1000 # approximate environmental adiabetic lapse rate for the air °C/m
                markers.id[mm] = 0 
                markers.T[mm] = 253 - dtdy*(ysize - markers.y[mm])

            # update marker index
            mm +=1
    plt.title('Model setup: initial markers \n shows different materials (air, bedrock, ice)')
    plt.scatter(markers.x, markers.y, c=markers.id)
    plt.xlim(0,xsize)
    plt.ylim(0,ysize)
    plt.xlabel('distance [m]')
    plt.ylabel('height [m]')
    plt.show()


    
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
        B_intern[4] = prescribed x-velocity value.
        B_intern[5] = y-index of vy nodes with prescribed velocity (-1 is not in use)
        B_intern[6-7] = min/max x-index of the wall
        B_intern[8] = prescribed y-velocity value.
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

    '''
    
    
    # instantiate a pre-populated parameters object
    params = Parameters()
    
    ## time 
    params.t_end = 1000 * 365.25 * 3600 * 24  # set end time at t = 100 years for now
    params.tstp_max = 365.25 * 3600 * 24 # max 1 year timestep
    params.ntstp_max = 360*2                    # maximum number of timesteps
    params.Temp_stp_max = 20                  # maximum number of temperature substeps
    
    # additional model options 
    # initial system size
    # xsize0 = 10000   # 10 km
    # ysize0 =  2000   # 2 km
    # xsize0 = 10000   # 4 km
    # ysize0 = 1100   # 1.5 km
    xsize0 = 4000
    ysize0 = 700
    
    xsize = xsize0
    ysize = ysize0

    # set resolution
    # xnum = 401   # 25 m
    # ynum = 81    # 25 m
    # xnum = 501#161   # 25 m
    # ynum = 76#61    # 25 m
    xnum = 161   # 25 m
    ynum = 61    # 25 m


    params.bx = xsize/(xnum-1)
    params.by = ysize/(ynum-1)
    params.Nx = 0
    params.Ny = 0
    params.non_uni_xsize = 0
    params.frict_yn = 0
    params.eta_min = 1e5
    params.eta_max = 1e25
    params.T_top = 273 - 20 - (6.5/1000)*ysize   # Top of model, temperature (air) is 20 degrees below zero
    params.T_bot = 273                           # Bottom of model, temperature (bedrock) is 2 degrees above zero
    

    # instantiate/load material properties object
    # file path must be from top directory (as that is where the fn is called from!)
    matData = np.loadtxt('./models/paleoclimate/material_properties.txt', skiprows=3, delimiter=",")
    materials = Materials(matData)
    
    
    # For the materials, we in fact only need 3? Air, ice and bedrock 
    

    # create directories for output of figures and data (not atm)
    pathlib.Path('./Figures').mkdir(exist_ok=True)
    pathlib.Path('./Output').mkdir(exist_ok=True)

    ###########################################################################    
    # Boundary conditions
    # pressure BCs
    P_first = np.array([0,1e5])

    # velocity BCs
    B_top = np.zeros((xnum+1,4))
    B_top[:,1] = 1

    B_bottom = np.zeros((xnum+1,4))
    B_bottom[:,1] = 1
    # B_bottom[:,2] = -params.v_ext/xsize * ysize

    B_left = np.zeros((ynum+1,4))
    B_left[:,3] = 1
    # B_left[:,0] = -params.v_ext/2
    # B_left[:,3] = 1

    B_right = np.zeros((ynum+1,4))
    B_right[:,3] = 1
    
    #B_right[:,0] = params.v_ext/2
    #B_right[:,3] = 1

    # optional internal boundary, switched off
    B_intern = np.zeros(8)
    B_intern[0] = -1
    B_intern[4] = -1
    
    # temperature BCs
    BT_top = np.zeros((xnum, 2))
    BT_bottom = np.zeros((xnum, 2))
    BT_left = np.zeros((ynum, 2))
    BT_right = np.zeros((ynum, 2))

    # upper and lower  = fixed T
    BT_top[:,0] = params.T_top
    BT_bottom[:,0] = params.T_bot

    # left and right = insulating?
    BT_left[:,1] = 1
    BT_right[:,1] = 1

    ###########################################################################
    # create grid object
    grid = Grid(xnum, ynum)

    # define grid points for (potentially) unevenly spaced grid
    gridSpacings(params.bx, params.by, params.Nx, params.Ny, params.non_uni_xsize, xsize, ysize, grid, 0)

    ############################################################################
    # create markers object
    mnumx = xnum*4#400
    mnumy = ynum*4#300
    markers = Markers(mnumx, mnumy)

    # initialize markers
    initialize_markers(markers, materials, params, xsize, ysize)
    
    return params, grid, materials, markers, xsize, ysize, P_first, B_top, B_bottom,\
           B_left, B_right, B_intern, BT_top, BT_bottom, BT_left, BT_right
    
    
    











    