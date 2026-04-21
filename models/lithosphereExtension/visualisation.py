#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualisation routines for plotting code output

"""
from matplotlib import figure
import numpy as np

from output.visualisation import getMarkerField, getMarkerPixelGrid, plotMarkers_lithology, plotSummary, plotTemperature

###############################################################################
# custom plotting routines

def plotMarkers_stress(params, markers, grid, ntstp, t_curr):
    '''
    Plot the stress components recorded by the markers.

    Parameters
    ----------
    params : Parameters object
        Simulation's parameters object.
    markers : Markers object
        Contains all the marker values for each variable.
    grid : Grid object
        Contains all the grid variables at the current time.
    ntstp : INT
        Current timestep number.
    t_curr : FLOAT
        Current time (s).

    Returns
    -------
    None.

    '''
    
    xlims = (grid.x[0], grid.x[-1])
    ylims = (grid.y[-1], grid.y[0])
    
    # get the mapping of markers to pixel positions
    marker_map = getMarkerPixelGrid(params, markers, grid, 401)
    
    # get the specific fields we want here
    mark_sigmaxx = getMarkerField(marker_map, markers.sigmaxx)
    mark_sigmaxy = getMarkerField(marker_map, markers.sigmaxy)
    mark_sigmaii = np.sqrt(mark_sigmaxx**2 + mark_sigmaxy**2)

    box_size = [xlims[0], xlims[1], ylims[0],ylims[1]]
    
    
    # create figure, subplots
    fig = figure.Figure(figsize=(18,18), constrained_layout=True)
    axs = fig.subplots(3,1, sharex=True)
    
    # create image grid and temperature contour levels
    X, Y = np.meshgrid(grid.x, grid.y)
    temp_levels = [100, 150, 350, 450, 1300]

    ###########################################################################
    # plot the stress
    im = axs[0].imshow(mark_sigmaii, origin='upper', aspect='auto', extent=box_size)             
    fig.colorbar(im, ax=axs[0],pad=0.0)
    axs[0].set_title('$\\sigma_{ii}$ (Pa)')
    axs[0].set(ylabel='y (m)', xlim=xlims, ylim=ylims)
    
    # add temperature contours
    cs = axs[0].contour(X, Y, grid.T-273, levels=temp_levels, colors='w', linewidths=0.8)
    axs[0].clabel(cs, inline=True, fontsize=8, fmt='%d C')                     
    
    ###########################################################################
    # plot normal stress components
    im = axs[1].imshow(mark_sigmaxx, origin='upper', aspect='auto', extent=box_size)
    fig.colorbar(im, ax=axs[1],pad=0.0) 
    axs[1].set_title('$\\sigma_{xx}$ (Pa)') 
    axs[1].set(ylabel = 'y (m)', xlim=xlims, ylim=ylims)
    
    # add temperature contours
    cs = axs[1].contour(X, Y, grid.T-273, levels=temp_levels, colors='w', linewidths=0.8)
    axs[1].clabel(cs, inline=True, fontsize=8, fmt='%d C')
    
    ###########################################################################
    # plot shear stress components
    im = axs[2].imshow(mark_sigmaxy, origin='upper', aspect='auto', extent=box_size)
    fig.colorbar(im, ax=axs[2],pad=0.0)     
    axs[2].set_title('$\\sigma_{xy}$ (Pa)')
    axs[2].set(xlabel='x (m)', ylabel = 'y (m)', xlim=xlims, ylim=ylims)   
    
    # add temperature contours
    cs = axs[2].contour(X, Y, grid.T-273, levels=temp_levels, colors='w', linewidths=0.8)
    axs[2].clabel(cs, inline=True, fontsize=8, fmt='%d C')

    fig.suptitle('Time: %.3f Myr'%(t_curr*1e-6/(365.25*24*3600)))
    fig.savefig('%s/%s/stress_%i.png'%(params.output_path, params.output_name, ntstp))




    
def plotMarkers_strain(params, markers, grid, ntstp, t_curr):
    '''
    Plot the strain components and accumulated strain recorded by the markers.

    Parameters
    ----------
    params : Parameters object
        Simulation's parameters object.
    markers : Markers object
        Contains all the marker values for each variable.
    grid : Grid object
        Contains all the grid variables at the current time.
    ntstp : INT
        Current timestep number.
    t_curr : FLOAT
        Current time (s).

    Returns
    -------
    None.

    '''
    
    # get the mapping of markers to pixel positions
    marker_map = getMarkerPixelGrid(params, markers, grid, 401)

    
    xlims = (grid.x[0], grid.x[-1])
    ylims = (grid.y[-1], grid.y[0])
    
    box_size = [xlims[0], xlims[1], ylims[0],ylims[1]]
    
    # create figure, subplots
    fig = figure.Figure(figsize=(18,18), constrained_layout=True)
    axs = fig.subplots(4,1, sharex=True)
    
    # create image grid and temperature contour levels
    X, Y = np.meshgrid(grid.x, grid.y)
    temp_levels = [100, 150, 350, 450, 1300]

    ###########################################################################
    # plot the normal strain rate components
    mark_epsxx = getMarkerField(marker_map, markers.epsxx)
    im = axs[0].imshow(mark_epsxx, origin='upper', aspect='auto', extent=box_size, vmin=-4e-14, vmax=4e-14)
    
    fig.colorbar(im, ax=axs[0],pad=0.0)
    axs[0].set_title('$\\dot\\epsilon_{xx}$ (1/s)')
    axs[0].set(ylabel='y (m)', xlim=xlims, ylim=ylims)
    
    #add temperature contours
    cs = axs[0].contour(X, Y, grid.T-273, levels=temp_levels, colors='w', linewidths=0.8)
    axs[0].clabel(cs, inline=True, fontsize=8, fmt='%d C')                     
    
    
    ###########################################################################
    # plot the shear strain rate components
    mark_epsxy = getMarkerField(marker_map, markers.epsxy)
    im = axs[1].imshow(mark_epsxy, origin='upper', aspect='auto', extent=box_size, vmin=-4e-14, vmax=4e-14)
    
    fig.colorbar(im, ax=axs[1],pad=0.0)
    axs[1].set_title('$\\dot\\epsilon_{xy}$ (1/s)')
    axs[1].set(ylabel = 'y (m)', xlim=xlims, ylim=ylims)
    
    # add temperature contours
    cs = axs[1].contour(X, Y, grid.T-273, levels=temp_levels, colors='w', linewidths=0.8)
    axs[1].clabel(cs, inline=True, fontsize=8, fmt='%d C')
    
    
    ###########################################################################
    # plot normal stress components
    mark_epsii = np.sqrt(mark_epsxy**2+mark_epsxy**2)
    im = axs[2].imshow(mark_epsii, origin='upper', aspect='auto', extent=box_size, vmin=-4e-14, vmax=4e-14)
    
    fig.colorbar(im, ax=axs[2],pad=0.0)
    axs[2].set_title('$\\dot \\epsilon_{ii}$ (1/s)')
    axs[2].set(ylabel='y (m)', xlim=xlims, ylim=ylims)
   
    # add temperature contours
    cs = axs[2].contour(X, Y, grid.T-273, levels=temp_levels, colors='w', linewidths=0.8)
    axs[2].clabel(cs, inline=True, fontsize=8, fmt='%d C')


    ###########################################################################
    # Plot accumulated strain
    mark_gii = getMarkerField(marker_map, markers.gII)
    im = axs[3].imshow(np.log10(mark_gii), origin='upper', aspect='auto', extent=box_size, vmin=-2, vmax=2)
    
    fig.colorbar(im, ax=axs[3],pad=0.0)
    axs[3].set_title('Total strain (log10)')
    axs[3].set(xlabel='x (m)', ylabel = 'y (m)', xlim=xlims, ylim=ylims) 

    # add temperature contours
    cs = axs[3].contour(X, Y, grid.T-273, levels=temp_levels, colors='w', linewidths=0.8)
    axs[3].clabel(cs, inline=True, fontsize=8, fmt='%d C')



    fig.suptitle('Time: %.3f Myr'%(t_curr*1e-6/(365.25*24*3600)))
    fig.savefig('%s/%s/strain_%i.png'%(params.output_path, params.output_name, ntstp))



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
    xlims = (grid.x[0], grid.x[-1])
    ylims = (grid.y[-1], grid.y[0])
    
    plotTemperature(grid, params, ntstp, t_curr, xlims, ylims, aspect_ratio=3)
    plotSummary(grid, params, ntstp, t_curr, xlims, ylims, aspect_ratio=3, plotTempContours=True, temp_levels=[100, 150, 350, 450, 1300])
    plotMarkers_lithology(params, markers, grid, ntstp, t_curr, xlims, ylims, aspect_ratio=3)
    plotMarkers_strain(params, markers, grid, ntstp, t_curr)
    plotMarkers_stress(params, markers, grid, ntstp, t_curr)
