#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

common functions that can be used by many models

"""

def uniformGrid(params, grid):
    '''
    Calculates the new grid spacings based on the current xsize and ysize.
    
    This default version implements a fixed, uniform grid.

    Parameters
    ----------
    params : Parameters Class
        Parameters object containing all simulation parameters for the system.
    grid : OBJ
        The grid object into which the new node positions will be written.
        
    Returns
    -------
    None.

    '''
    
    
    xnum = grid.xnum
    ynum = grid.ynum
    
    dx = params.xsize/(xnum-1)
    dy = params.ysize/(ynum-1)
    
    # Simple, uniform grid
    # horizontal grid
    grid.x[0] = 0
    for i in range(1,xnum):
        grid.x[i] = grid.x[i-1] + dx
        
    # vertical grid
    grid.y[0] = 0
    for i in range(1,ynum):
        grid.y[i] = grid.y[i-1] + dy