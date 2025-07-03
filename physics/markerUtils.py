#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Low level functions for marker calculations, including marker-to-grid/grid-to-marker interpolation, and finding nearest node distances.

"""
import numpy as np
from dataStructures import Markers, Materials, Grid
from numba import jit

@jit(nopython=True)
def applyMarkerContrib(field, m_field, dxm, dym, j, i, mwt, width=-1.0):
    '''
    Applies the contribution to given field from a marker to the
    surrounding nodes based on the distance to the nodes.

    Parameters
    ----------
    field : ARRAY
        The grid variable to be added to.
    m_field : FLOAT
        The marker value for the variable provided.
    weights : ARRAY
        The weighting functions for each of the surrounding nodes.
    i : INT
        y-index of the top-left node.
    j : INT
        x-index of the top-left node.
    dx : FLOAT
        fractional distance from top-left node within cell in x-direction.
    dy : FLOAT
        fractional distance from top-left node within cell in y-direction.
    mwt : FLOAT
        Additional marker weighting value.
    width : FLOAT (Default=-1)
        The cell width around the node from which to apply the contribution, 
        if less than the whole zone.  Default is -1, which doesn't use this limiting.
        

    Returns
    -------
    None.

    '''
    if (width > 0):
        if (dxm <= width and dym <= width):
            # i,j node
            field[i,j] += m_field*(1-dxm)*(1-dym)*mwt
        if (dxm <= width and dym >= width):
            # i+1, j node
            field[i+1,j] += m_field*(1-dxm)*dym*mwt
        if (dxm >= width and dym <= width):
            # i, j+1 node
            field[i,j+1] += m_field*dxm*(1-dym)*mwt
        if (dxm >= width and dym >= width):
            # i+1, j+1 node
            field[i+1,j+1] += m_field*dxm*dym*mwt
    else:
        field[i,j] += m_field*(1-dxm)*(1-dym)*mwt
        field[i+1,j] += m_field*(1-dxm)*dym*mwt
        field[i,j+1] += m_field*dxm*(1-dym)*mwt
        field[i+1,j+1] += m_field*dxm*dym*mwt

@jit(nopython=True)
def applyGridContrib(field, xn, yn, dxm, dym):
    '''
    Applies the contributions for the surrounding grid nodes to a marker for a 
    given variable.

    Parameters
    ----------
    field : ARRAY
        The grid values of the required variable.
    xn : INT
        x-index of the top-left node.
    yn : INT
        y-index of the top-left node.
    dxm : FLOAT
        fractional distance from top-left node within cell in x-direction.
    dym : FLOAT
        fractional distance from top-left node within cell in y-direction. T

    Returns
    -------
    fm : FLOAT
        The calulated marker value for the variable.

    '''
    # add contibutions for 4 surrounding nodes
    fm = (1-dxm)*(1-dym)*field[yn, xn]
    fm += (1-dxm)*dym*field[yn+1, xn]
    fm += dxm*(1-dym)*field[yn, xn+1]
    fm += dxm*dym*field[yn+1, xn+1]

    return fm

@jit(nopython=True)    
def getMarkerNodeDistances(markerx, markery, markernx, markerny, xnum, ynum, grid, node_type):
    '''
    Finds the distance to the nearest top-left node of a specified type for a given marker position.

    Parameters
    ----------
    markerx : FLOAT
        x-coord of the marker.
    markery : FLOAT
        y-coord of the marker.
    markernx : INT
        Last recorded nearest basic node x-index.
    markerny : INT
        Last recorded nearest basic node y-index.
    xnum : INT
        x resolution of grid.
    ynum : INT
        y resolution of grid.
    grid : Grid
        The grid object containing all grid variables.
    node_type : INT
        Which type of node to use. 0 = basic nodes, 1 = pressure nodes

    Returns
    -------
    dxm : FLOAT
        x-distance to the top-left node.
    dym : FLOAT
        y-distance to the top-left node.
    xn : INT
        x-index of the top-left node.
    yn : INT
        y-index of the top-left node.

    '''
    
    # get indicies of top-left node
    xn = markernx
    yn = markerny
    
    # if we are interpolating from pressure nodes, need to get correct dx, dy
    if (node_type==1):
        # pressure node
        # horizontal index
        if (markerx < grid.cx[xn+1]):
            xn = xn - 1
    
        if (xn<0):
            xn = 0
        
        elif(xn>xnum-3):
            xn = xnum - 3
        
        # vertical index
        if (markery < grid.cy[yn+1]):
            yn = yn - 1
    
        if (yn<0):
            yn = 0
        
        elif(yn>ynum-3):
            yn = ynum - 3
        
    
    # define the normalised distance from node
    if (node_type==1):
        # pressure nodes, use centered spacings
        dxm = (markerx - grid.cx[xn+1])/grid.xstpc[xn+1]
        dym = (markery - grid.cy[yn+1])/grid.ystpc[yn+1]
    else:
        dxm = (markerx - grid.x[xn])/grid.xstp[xn]
        dym = (markery - grid.y[yn])/grid.ystp[yn]

    return dxm, dym, xn, yn

@jit(nopython=True)
def findNearestNode(gridx, gridy, xnum, ynum, markerx, markery):
    '''
    Find the index of the nearest top-left basic node.

    Parameters
    ----------
    gridx : ARRAY
        x-positions of the basic nodes.
    gridy : ARRAY
        y-positions of the basic nodes.
    xnum : INT
        x-resolution of the simulation domain.
    ynum : INT
        y-resolution of the simulation domain.
    markerx : FLOAT
        x-coord of the marker.
    markery : FLOAT
        y-coord of the marker.

    Returns
    -------
    xn : INT
        x-index of the top-left node.
    yn : INT
        y-index of the top-left node.

    '''
    
    # horizontal
    xnmin = 0
    xnmax = xnum-1
    while (xnmax-xnmin>1):
        xn = int((xnmax+xnmin)/2)
        # check if our marker is in the upper or lower half of the current range
        if (gridx[xn]>markerx):
            # take the new range as the lower half
            xnmax = xn
        else:
            # take the new rnage as the upper half
            xnmin = xn
    
    # xnmin now the closest node to the left of marker
    xn = xnmin
    
    if (xn<0):
        xn = 0
    elif (xn>xnum-2):
        xn = xnum-2

    
    # do the same for the vertical index
    ynmin = 0
    ynmax = ynum-1
    while (ynmax-ynmin>1):
        yn = int((ynmax+ynmin)/2)
        # check if our marker is in the upper or lower half of the current range
        if (gridy[yn]>markery):
            # take the new range as the lower half
            ynmax = yn
        else:
            # take the new rnage as the upper half
            ynmin = yn
    
    # xnmin now the closest node to the left of marker
    yn = ynmin
    
    if (yn<0):
        yn = 0
    elif (yn>ynum-2):
        yn = ynum-2

    return xn, yn


@jit(nopython=True) 
def applyGridWeights(xnum, ynum, grid, grid0):
    '''
    Apply the weighting to the averaging of the grid values, 
    to complete the interpolation from markers in markersToGrid.

    Parameters
    ----------
    xnum : INT
        x-resolution of the simulation domain.
    ynum : INT
        y-resolution of the simulation domain.
    grid : Grid
        Grid object containing all the grid variables, will be updated by this function.
    grid0 : Grid
        Grid object containing all the previous step's grid variables.

    Returns
    -------
    None.

    '''
    
    # have to loop to check that weight != 0
    for i in range(0,ynum):
        for j in range(0,xnum):
            if (grid.wt[i,j] >= 1e-7):
                grid.rho[i,j] = grid.rho[i,j]/grid.wt[i,j]
                grid.rhoCP[i,j] = grid.rhoCP[i,j]/grid.wt[i,j]
                grid.T[i,j] = grid.T[i,j]/grid.wt[i,j]
                grid.kT[i,j] = grid.kT[i,j]/grid.wt[i,j]
                grid.H_a[i,j] = grid.H_a[i,j]/grid.wt[i,j]
                grid.H_r[i,j] = grid.H_r[i,j]/grid.wt[i,j]
                
            else:
                # no new values interpolated, use previous
                grid.rho[i,j] = grid0.rho[i,j]
                grid.rhoCP[i,j] = grid0.rhoCP[i,j]
                grid.T[i,j] = grid0.T[i,j]
                grid.kT[i,j] = grid0.kT[i,j]
                grid.H_a[i,j] = grid0.H_a[i,j]
                grid.H_r[i,j] = grid0.H_r[i,j]
                
            # shear visc
            if (grid.wt_eta_s[i,j] >= 1e-7):
                grid.eta_s[i,j] = grid.eta_s[i,j]/grid.wt_eta_s[i,j]
                # this one is inverted, revert it!
                grid.mu_s[i,j] = 1/(grid.mu_s[i,j]/grid.wt_eta_s[i,j])
                grid.sigxy[i,j] = grid.sigxy[i,j]/grid.wt_eta_s[i,j]
            else:
                grid.eta_s[i,j] = grid0.eta_s[i,j]
                grid.mu_s[i,j] = grid0.mu_s[i,j]
                grid.sigxy[i,j] = grid0.sigxy[i,j]
            
            # normal visc
            if (i<ynum-1 and j<xnum-1):
                if (grid.wt_eta_n[i,j] >= 1e-7):    
                    grid.eta_n[i,j] = grid.eta_n[i,j]/grid.wt_eta_n[i,j]
                    # this one is inverted, revert it!
                    grid.mu_n[i,j] = 1/(grid.mu_n[i,j]/grid.wt_eta_n[i,j])
                    grid.sigxx[i,j] = grid.sigxx[i,j]/grid.wt_eta_n[i,j]
                else:
                    grid.eta_n[i,j] = grid0.eta_n[i,j]
                    grid.mu_n[i,j] = grid0.mu_n[i,j]
                    grid.sigxx[i,j] = grid0.sigxx[i,j]
           
