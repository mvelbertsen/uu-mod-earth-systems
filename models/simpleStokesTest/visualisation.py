#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualisation routines for plotting code output

"""

from output.visualisation import plotSummary, plotMarkers_lithology 


def makePlots(grid, markers, params, ntstp, t_curr):
    """
    Wrapper function which calls all plotting routines, to simplify calling in 
    the run script.

    Parameters
    ----------
    grid : grid Object
        grid object containing the all the simulation variables on the grid.
    markers : Markers object
        markers object containing the current marker quantities.
    params : Parameters object
        Object containing parameters for the simulation.
    ntstp: INT
        Current timestep number
    t_curr : FLOAT
        Current simulation time.

    Returns
    -------
    None.

    """
    xlims = (0,params.xsize)
    ylims = (params.ysize,0)
    
    
    plotSummary(grid, params, ntstp, t_curr, xlims, ylims)
    plotMarkers_lithology(params, markers, grid, ntstp, t_curr, xlims, ylims, height=6)
    
    
