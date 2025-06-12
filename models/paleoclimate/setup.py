#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File which contains the setup for a simple glacier model

"""

import numpy as np
import pathlib

import matplotlib.pyplot as plt

from dataStructures import Markers, Parameters, Grid, Materials
from physics.grid_fns import gridSpacings


###### VERSION 1 ######

def mountain_slope_curve(x, ysize):
    '''
    Define bedrock slope as straight line at slope of 10%.

    '''

    b = ysize/2                                # vertical offset
    mountain_slope = -0.1*x + b

    return mountain_slope


def glacier_surface_curve(x, ysize):
    '''
    Define glacier surface slope as concave down decreasing curve.

    '''

    b = ysize/8*5                              # vertical offset
    glacier_surface = -0.1*x - 2e-5*x**2 + b

    return glacier_surface


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
            
            # now define type based on location, to give our initial setup
            # sticky air
            markers.id[mm] = 0
            markers.T[mm]  = 273        # 0 °C
            
            # bedrock
            if (markers.y[mm] <= mountain_slope_curve(markers.x[mm], ysize)):
                markers.id[mm] = 1
                markers.T[mm]  = 273    # 0 °C
            
            # glacier   # try to reproduce glacier shape from SIA
            if (markers.y[mm] > mountain_slope_curve(markers.x[mm], ysize) and markers.y[mm] <= glacier_surface_curve(markers.x[mm], ysize)+750.): ### !!! voor nu + 750 om grotere glacier te krijgen met lagere res.
                markers.id[mm] = 2
                markers.T[mm]  = 263    # -10 °C          
            
            # update marker index
            mm +=1

    plt.title('Markers initial: shows different materials (air, bedrock, ice)')
    plt.scatter(markers.x, markers.y, c=markers.id)
    plt.xlim(0,xsize)
    plt.ylim(0,ysize)
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

    # adapt params to change time
    params.t_end = 360*365.25*24*3600           # end time
    params.tstp_max = 365.25*24*3600            # maximum timestep (1 year)

    # additional model options 
    # initial system size
    xsize0 = 10000   # 10 km
    ysize0 = 2000    #  2 km
    
    xsize = xsize0
    ysize = ysize0

    # set resolution (number of markers in system size) ### !!! voor nu laag, straks hoger zetten --> hoger geeft ZeroDivisionError
    xnum = 16#101       # 100 m
    ynum = 21#21        # 100 m

    # adapt params to create uniform grid
    params.bx = xsize/(xnum-1)               # x-grid spacing in high res area
    params.by = ysize/(ynum-1)               # y-grid spacing in high res area
    params.Nx = 0                            # number of unevenly spaced grid points either side of high res zone
    params.Ny = 0                            # number of unevenly spaced grid points below high res zone
    params.non_uni_xsize = 0                 # physical x-size of non-uniform grid region
    params.v_ext = 0

    # instantiate/load material properties object
    # file path must be from top directory (as that is where the fn is called from!)
    matData = np.loadtxt('./models/paleoclimate/material_properties.txt', skiprows=3, delimiter=",")
    materials = Materials(matData)

    # create directories for output of figures and data (not atm)
    pathlib.Path('./Figures').mkdir(exist_ok=True)
    pathlib.Path('./Output').mkdir(exist_ok=True)

    ###########################################################################    
    # Boundary conditions
    # pressure BCs
    P_first = np.array([1,1e5]) # [0] sets type of P BC (0=defined by one cell, 1= top and bottom), [1] sets boundary value

    # velocity BCs
    B_top = np.zeros((xnum+1,4))
    B_top[:,1] = 1                    # free slip

    B_bottom = np.zeros((xnum+1,4))
    # B_bottom[:,1] = 1               # no slip (turn on for free slip --> basal sliding)

    B_left = np.zeros((ynum+1,4))
    B_left[:,3] = 1                   # free slip

    B_right = np.zeros((ynum+1,4))
    B_right[:,0] = 0                  # change to v inflow vel. to equal deform ???

    # optional internal boundary, switched off
    B_intern = np.zeros(8)
    B_intern[0] = -1
    B_intern[4] = -1
    
    # temperature BCs
    BT_top = np.zeros((xnum, 2))
    BT_bottom = np.zeros((xnum, 2))
    BT_left = np.zeros((ynum, 2))
    BT_right = np.zeros((ynum, 2))

    # set parameters
    params.T_top = 253.
    params.T_bot = 283.

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
    mnumx = 101#400
    mnumy = 21#300
    markers = Markers(mnumx, mnumy)

    # initialize markers
    initialize_markers(markers, materials, params, xsize, ysize)
    
    return params, grid, materials, markers, xsize, ysize, P_first, B_top, B_bottom,\
           B_left, B_right, B_intern, BT_top, BT_bottom, BT_left, BT_right






###### VERSION 2 ######

# def mountain_slope_curve(x, xsize, ysize):
#     '''
#     Define bedrock slope as concave up decreasing curve. Follows roughly bedrock slope of 10% (y=0.1x).

#     '''

#     a = (xsize/ysize*0.1) * ysize/(xsize**2)   # coefficient
#     b = ysize/5                                # vertical offset
#     mountain_slope = a * (x-xsize)**2 + b      # note horizontal shift along x-axis (of -xsize)

#     return mountain_slope


# def glacier_surface_curve(x, xsize, ysize):
#     '''
#     Define glacier surface slope as concave down decreasing curve.

#     '''

#     a =  -(xsize/ysize*0.1) * ysize/((xsize/10*8)**2)   # coefficient
#     b = xsize/10*0.75 + ysize/5                         # vertical offset
#     glacier_surface = a * x**2 + b

#     return glacier_surface


# def initialize_markers(markers, materials, params, xsize, ysize):
#     '''
#     Initialize the positions, material ID and temperature of the markers.

#     Parameters
#     ----------
#     markers : Markers Object
#         An empty markers object, which will be filled by this function.
#     materials : Materials object
#         Materials object filled from file with the required materials.
#     params : Parameters Object
#         Contains the simulation parameters
#     xsize : FLOAT
#         Physical x-size of the simulation domain.
#     ysize : FLOAT
#         Physical y-size of the simulation domain.

#     Returns
#     -------
#     None.

#     '''

#     np.random.seed(1337)
#     # set the rough grid spacing for the markers
#     mxstp = xsize/markers.xnum
#     mystp = ysize/markers.ynum
    
#     # marker number index
#     mm = 0

#     for j in range(0, markers.xnum):
#         for i in range(0, markers.ynum):
#             # set coordinates as grid + small random displacement
#             markers.x[mm] = (j + np.random.random())*mxstp
#             markers.y[mm] = (i + np.random.random())*mystp
            
#             # now define type based on location, to give our initial setup
#             # sticky air
#             markers.id[mm] = 0
#             markers.T[mm]  = 273        # 0 °C
#             # markers.T[mm] = params.T_top  # Set initial temperature for simplicity
            
#             # bedrock
#             if markers.y[mm] <= mountain_slope_curve(markers.x[mm], xsize, ysize):
#                 markers.id[mm] = 1
#                 markers.T[mm]  = 273    # 0 °C
#                 # markers.T[mm] = params.T_top  # Set initial temperature for simplicity
            
#             # glacier   # try to reproduce glacier shape from SIA
#             elif (markers.y[mm] > mountain_slope_curve(markers.x[mm], xsize, ysize) and markers.y[mm] <= glacier_surface_curve(markers.x[mm], xsize, ysize)):
#                 markers.id[mm] = 2
#                 markers.T[mm]  = 263    # -10 °C
#                 # markers.T[mm] = params.T_top-10.  # Set initial temperature for simplicity

#             # update marker index
#             mm += 1
    
#     plt.title('Markers initial: shows different materials (air, bedrock, ice)')
#     plt.scatter(markers.x, markers.y, c=markers.id)
#     plt.xlim(0,xsize)
#     plt.ylim(0,ysize)
#     plt.show()



# def initializeModel():
#     '''
#     Sets up the initial state of the model, including BCs and output settings.

#     Returns
#     -------
#     params : Parameters object
#         Model physical and numerical parameters.
#     grid : Grid object
#         Initialised Grid object.
#     materials : Materials
#         Materials object initialsed with required material properties.
#     markers : Markers
#         Initialized markers object.
#     xsize : INT
#         x-resolution of model domain.
#     ysize : INT
#         y-resolution of model domain.
#     P_first : ARRAY
#         Array with 2 entries, specifying pressure BC.
#     B_top : ARRAY
#         Boundary conditions at the top of the grid. Array has 4 columns, 
#         values in each are defined as: 
#         vx[0,j] = B_top[j,0] + vx[1,j]*B_top[j,1]
#         vy[0,j] = B_top[j,2] + vy[1,j]*B_top[j,3]
#     B_bottom : ARRAY
#         Boundary conditions at the bottom of the grid. Array has 4 columns, 
#         values in each are defined as: 
#         vx[0,j] = B_bot[j,0] + vx[1,j]*B_bot[j,1]
#         vy[0,j] = B_bot[j,2] + vy[1,j]*B_bot[j,3]
#     B_left : ARRAY
#         Boundary conditions at the left of the grid. Array has 4 columns, 
#         values in each are defined as: 
#         vx[0,i] = B_left[i,0] + vx[1,i]*B_left[i,1]
#         vy[0,i] = B_left[j,2] + vy[1,i]*B_left[i,3]
#     B_right : ARRAY
#         Boundary conditions at the left of the grid. Array has 4 columns, 
#         values in each are defined as: 
#         vx[0,i] = B_right[i,0] + vx[1,i]*B_right[i,1]
#         vy[0,i] = B_right[j,2] + vy[1,i]*B_right[i,3]
#     B_intern : ARRAY
#         Array defining optional internal boundary eg. moving wall. Format is:
#         B_intern[0] = x-index of vx nodes with prescribed velocity (-1 is not in use)
#         B_intern[1-2] = min/max y-index of the wall
#         B_intern[4] = prescribed x-velocity value.
#         B_intern[5] = y-index of vy nodes with prescribed velocity (-1 is not in use)
#         B_intern[6-7] = min/max x-index of the wall
#         B_intern[8] = prescribed y-velocity value.
#     BT_top : ARRAY
#         Top temperature BCs.  Array has 2 columns, values in each are defined as:
#         T[i,j] = BT_top[0] + BT_top[1]*T[i+1,j]
#     BT_bottom : ARRAY
#         Bottom temperature BCs.  Array has 2 columns, values in each are defined as:
#         T[i,j] = BT_bottom[0] + BT_bottom[1]*T[i-1,j]
#     BT_left : ARRAY
#         Left temperature BCs.  Array has 2 columns, values in each are defined as:
#         T[i,j] = BT_left[0] + BT_left[1]*T[i,j+1]
#     BT_right : ARRAY
#         Right temperature BCs.  Array has 2 columns, values in each are defined as:
#         T[i,j] = BT_right[0] + BT_right[1]*T[i,j-1]

#     '''
    
#     # instantiate a pre-populated parameters object
#     params = Parameters()

#     # adapt params to change time
#     params.t_end = 360*365.25*24*3600           # end time
#     params.tstp_max = 365.25*24*3600            # maximum timestep (1 year)
#     # params.t_end = 72e3                       # end time

#     # additional model options 
#     # initial system size
#     xsize0 = 10000   # 10 km
#     ysize0 = 2000    #  2 km
    
#     xsize = xsize0
#     ysize = ysize0

#     # set resolution (number of markers in system size)
#     xnum = 101
#     ynum = 51

#     # adapt params to create uniform grid
#     params.bx = xsize/(xnum-1)               # x-grid spacing in high res area
#     params.by = ysize/(ynum-1)               # y-grid spacing in high res area
#     params.Nx = 0                            # number of unevenly spaced grid points either side of high res zone
#     params.Ny = 0                            # number of unevenly spaced grid points below high res zone
#     params.non_uni_xsize = 0                 # physical x-size of non-uniform grid region
#     params.v_ext = 0

#     # instantiate/load material properties object
#     # file path must be from top directory (as that is where the fn is called from!)
#     matData = np.loadtxt('./models/paleoclimate/material_properties.txt', skiprows=3, delimiter=",")
#     materials = Materials(matData)

#     # create directories for output of figures and data (not atm)
#     pathlib.Path('./Figures').mkdir(exist_ok=True)
#     pathlib.Path('./Output').mkdir(exist_ok=True)

#     ###########################################################################    
#     # Boundary conditions
#     # pressure BCs
#     P_first = np.array([1,1e5]) # [0] sets type of P BC (0=defined by one cell, 1= top and bottom), [1] sets boundary value

#     # velocity BCs
#     B_top = np.zeros((xnum+1,4))
#     B_top[:,1] = 1                    # free slip

#     B_bottom = np.zeros((xnum+1,4))
#     # B_bottom[:,1] = 1               # no slip (turn on for free slip --> basal sliding)

#     B_left = np.zeros((ynum+1,4))

#     B_right = np.zeros((ynum+1,4))

#     # optional internal boundary, switched off
#     B_intern = np.zeros(8)
#     B_intern[0] = -1
#     B_intern[4] = -1
    
#     # temperature BCs
#     BT_top = np.zeros((xnum, 2))
#     BT_bottom = np.zeros((xnum, 2))
#     BT_left = np.zeros((ynum, 2))
#     BT_right = np.zeros((ynum, 2))

#     # set parameters
#     params.T_top = 253.
#     params.T_bot = 283.
#     # params.T_bot = 275.

#     # upper and lower  = fixed T
#     BT_top[:,0] = params.T_top
#     BT_bottom[:,0] = params.T_bot
#     # BT_bottom[:,0] = params.T_top

#     # left and right = insulating?
#     BT_left[:,1] = 1
#     BT_right[:,1] = 1

#     ###########################################################################
#     # create grid object
#     grid = Grid(xnum, ynum)

#     # define grid points for (potentially) unevenly spaced grid
#     gridSpacings(params.bx, params.by, params.Nx, params.Ny, params.non_uni_xsize, xsize, ysize, grid, 0)

#     ############################################################################
#     # create markers object
#     mnumx = 101
#     mnumy = 51
#     markers = Markers(mnumx, mnumy)

#     # initialize markers
#     initialize_markers(markers, materials, params, xsize, ysize)
    
#     return params, grid, materials, markers, xsize, ysize, P_first, B_top, B_bottom,\
#            B_left, B_right, B_intern, BT_top, BT_bottom, BT_left, BT_right