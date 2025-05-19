#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple intrusion with visc, T,  density contrast
"""
import numpy as np

from dataStructures import Markers, Parameters, Grid, Materials
from physics.grid_fns import gridSpacings
import pathlib

def initializeModel():
    
    # function which should contain everything required to set up a simulation
    # so that we can easily swap out setups
    
    # instantiate a pre-populated parameters object
    params = Parameters()
    
    params.v_ext = 0.0
    

    # additional model options 
    # initial system size
    xsize0 = 1e5
    ysize0 = 1.5e5
    
    xsize = xsize0
    ysize = ysize0

    # set resolution
    xnum = 31
    ynum = 21
    
    params.bx = xsize/(xnum-1)
    params.by = ysize/(ynum-1)
    params.Nx = 0
    params.Ny = 0
    params.non_uni_xsize = 0


    # instantiate/load material properties object
    matData = np.loadtxt('./models/simpleStokesTest/material_properties_simple.txt', skiprows=3, delimiter=",")
    materials = Materials(matData) 

    params.save_fig = 2
    params.ntstp_max = 10

    # create directories for output of figures and data (not atm)
    pathlib.Path('./Figures').mkdir(exist_ok=True)
    pathlib.Path('./Output').mkdir(exist_ok=True)

    ###########################################################################    
    # Boundary conditions
    # pressure BCs
    P_first = np.array([0,1e5])

    # velocity BCs
    B_top = np.zeros([xnum+1,4])
    B_top[:,1] = 1

    B_bottom = np.zeros([xnum+1,4])
    B_bottom[:,1] = 1

    B_left = np.zeros([ynum+1,4])
    B_left[:,3] = 1

    B_right = np.zeros([ynum+1,4])
    B_right[:,3] = 1

    # optional internal boundary, switched off
    B_intern = np.zeros([8])
    B_intern[0] = -1
    B_intern[4] = -1
    
    # temperature BCs
    BT_top = np.zeros([xnum, 2])
    BT_bottom = np.zeros([xnum, 2])
    BT_left = np.zeros([ynum, 2])
    BT_right = np.zeros([ynum, 2])

    # upper and lower  - symmetry
    BT_top[:,1] = 1
    BT_bottom[:,1] = 1

    # left and right = symmetry/insulating? = -1?
    BT_left[:,1] = 1
    BT_right[:,1] = 1

    ###########################################################################
    # create grid object
    grid = Grid(xnum, ynum)
    

    # define grid points for (potentially) unevenly spaced grid
    gridSpacings(params.bx, params.by, params.Nx, params.Ny, params.non_uni_xsize, xsize, ysize, grid, 0)

    ############################################################################
    # create markers object
    mnumx = 400
    mnumy = 300
    markers = Markers(mnumx, mnumy)

    # initialize markers
    initialize_markers(markers, materials, params, xsize, ysize)
    
    return params, grid, materials, markers, xsize, ysize, P_first, B_top, B_bottom,\
           B_left, B_right, B_intern, BT_top, BT_bottom, BT_left, BT_right
           
           

def initialize_markers(markers, materials, params, xsize, ysize):
    
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
            
            # now define rock type based on location, to give our initial setup
            # inside the intrusion
            if (markers.y[mm] > ysize*0.4 and markers.y[mm] < ysize*0.6 and markers.x[mm] > xsize*0.4 and markers.x[mm] < xsize*0.6):
                markers.id[mm] = 0
                markers.T[mm] = 1300
            else:
                markers.id[mm] = 1
                markers.T[mm] = 1000
                
            
            # update marker index
            mm +=1
