#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualisation routines for plotting code output

"""
from matplotlib import figure
import numpy as np

from output.visualisation import getMarkerField, getMarkerPixelGrid, plotMarkers_lithology, plotSummary, plotTemperature

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
    temp_levels = [-2, -1, 0, 1, 2, 5, 10, 100, 500]

    ###########################################################################
    # plot the stress
    im = axs[0].imshow(mark_sigmaii, origin='upper', aspect='auto', extent=box_size, vmin=0, vmax=1.75e5)             
    fig.colorbar(im, ax=axs[0],pad=0.0, extend='both')
    axs[0].set_title('$\\sigma_{ii}$ (Pa)')
    axs[0].set(ylabel='y (m)', xlim=xlims, ylim=ylims)
    
    # add temperature contours
    cs = axs[0].contour(X, Y, grid.T-273, levels=temp_levels, colors='w', linewidths=0.8)
    axs[0].clabel(cs, inline=True, fontsize=8, fmt='%d C')                     
    
    ###########################################################################
    # plot normal stress components
    im = axs[1].imshow(mark_sigmaxx, origin='upper', aspect='auto', extent=box_size, vmin=0, vmax=1e5)
    fig.colorbar(im, ax=axs[1],pad=0.0, extend='both') 
    axs[1].set_title('$\\sigma_{xx}$ (Pa)') 
    axs[1].set(ylabel = 'y (m)', xlim=xlims, ylim=ylims)
    
    # add temperature contours
    cs = axs[1].contour(X, Y, grid.T-273, levels=temp_levels, colors='w', linewidths=0.8)
    axs[1].clabel(cs, inline=True, fontsize=8, fmt='%d C')
    
    ###########################################################################
    # plot shear stress components
    im = axs[2].imshow(mark_sigmaxy, origin='upper', aspect='auto', extent=box_size, vmin=-4e4, vmax=3e4)
    fig.colorbar(im, ax=axs[2],pad=0.0, extend='both')     
    axs[2].set_title('$\\sigma_{xy}$ (Pa)')
    axs[2].set(xlabel='x (m)', ylabel = 'y (m)', xlim=xlims, ylim=ylims)   
    
    # add temperature contours
    cs = axs[2].contour(X, Y, grid.T-273, levels=temp_levels, colors='w', linewidths=0.8)
    axs[2].clabel(cs, inline=True, fontsize=8, fmt='%d C')

    fig.suptitle('Time: %.3f yr'%(t_curr/(365.25*24*3600)))
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
    temp_levels = [-2, -1, 0, 1, 2, 5, 10, 100, 500]

    ###########################################################################
    # plot the normal strain rate components
    mark_epsxx = getMarkerField(marker_map, markers.epsxx)
    im = axs[0].imshow(mark_epsxx, origin='upper', aspect='auto', extent=box_size, vmin=-2.5e-9, vmax=2.5e-9)
    
    fig.colorbar(im, ax=axs[0],pad=0.0, extend='both')
    axs[0].set_title('$\\dot\\epsilon_{xx}$ (1/s)')
    axs[0].set(ylabel='y (m)', xlim=xlims, ylim=ylims)
    
    #add temperature contours
    cs = axs[0].contour(X, Y, grid.T-273, levels=temp_levels, colors='w', linewidths=0.8)
    axs[0].clabel(cs, inline=True, fontsize=8, fmt='%d C')                     
    
    
    ###########################################################################
    # plot the shear strain rate components
    mark_epsxy = getMarkerField(marker_map, markers.epsxy)
    im = axs[1].imshow(mark_epsxy, origin='upper', aspect='auto', extent=box_size, vmin=-5.5e-9, vmax=2e-9)
    
    fig.colorbar(im, ax=axs[1],pad=0.0, extend='both')
    axs[1].set_title('$\\dot\\epsilon_{xy}$ (1/s)')
    axs[1].set(ylabel = 'y (m)', xlim=xlims, ylim=ylims)
    
    # add temperature contours
    cs = axs[1].contour(X, Y, grid.T-273, levels=temp_levels, colors='w', linewidths=0.8)
    axs[1].clabel(cs, inline=True, fontsize=8, fmt='%d C')
    
    
    ###########################################################################
    # plot normal stress components
    mark_epsii = np.sqrt(mark_epsxy**2+mark_epsxy**2)
    im = axs[2].imshow(mark_epsii, origin='upper', aspect='auto', extent=box_size, vmin=0e-9, vmax=8e-9)
    
    fig.colorbar(im, ax=axs[2],pad=0.0, extend='both')
    axs[2].set_title('$\\dot \\epsilon_{ii}$ (1/s)')
    axs[2].set(ylabel='y (m)', xlim=xlims, ylim=ylims)
   
    # add temperature contours
    cs = axs[2].contour(X, Y, grid.T-273, levels=temp_levels, colors='w', linewidths=0.8)
    axs[2].clabel(cs, inline=True, fontsize=8, fmt='%d C')


    ###########################################################################
    # Plot accumulated strain
    mark_gii = getMarkerField(marker_map, markers.gII)
    im = axs[3].imshow(np.log10(mark_gii), origin='upper', aspect='auto', extent=box_size, vmin=-5, vmax=1.5)
    
    fig.colorbar(im, ax=axs[3],pad=0.0, extend='both')
    axs[3].set_title('Total strain (log10)')
    axs[3].set(xlabel='x (m)', ylabel = 'y (m)', xlim=xlims, ylim=ylims) 

    # add temperature contours
    cs = axs[3].contour(X, Y, grid.T-273, levels=temp_levels, colors='w', linewidths=0.8)
    axs[3].clabel(cs, inline=True, fontsize=8, fmt='%d C')

    fig.suptitle('Time: %.3f yr'%(t_curr/(365.25*24*3600)))
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
    plotSummary(grid, params, ntstp, t_curr, xlims, ylims, aspect_ratio=3, plotTempContours=True, temp_levels=[-10, -5, 0, 5, 10, 25, 50, 100, 500])
    plotMarkers_lithology(params, markers, grid, ntstp, t_curr, xlims, ylims, aspect_ratio=3)
    plotMarkers_strain(params, markers, grid, ntstp, t_curr)
    plotMarkers_stress(params, markers, grid, ntstp, t_curr)
    Plot_Vis_strain_stress(params, markers, grid, ntstp, t_curr, xlims, ylims, aspect_ratio=3, plotTempContours=True, temp_levels=[-10, -5, 0, 5, 10, 25, 50, 100, 500])

def Plot_Vis_strain_stress(params, markers, grid, ntstp, t_curr, xlims, ylims, aspect_ratio=3, plotTempContours=True, temp_levels=[-10, -5, 0, 5, 10, 25, 50, 100, 500]):

    '''
    Plot the stress and strain components recorded by the markers. Viscosity recorded on the grid

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
    # For the viscosity plot
    X, Y = np.meshgrid(grid.x, grid.y)
    
    # get aspect ratio
    xlen = abs(xlims[1] - xlims[0])
    ylen = abs(ylims[1] - ylims[0])
    
    if aspect_ratio == None:
        asp_rat = xlen/ylen
    else:
        asp_rat = aspect_ratio
    
    if asp_rat <= 1:
        # add some padding bc. of axis/cb taking up space in x
        pad = 2.0
    else:
        pad = 0.0
    
    
    # set ysize of each figure
    y_fig = 6
    
    # number of plots vertically
    y_plots = 3
    
    # create figure
    fig = figure.Figure(figsize=(asp_rat*y_fig + pad, y_fig*y_plots), constrained_layout=True)
    axs = fig.subplots(3,1, sharex=True, sharey=True)
    
    # check that if temp contours are switched on, values have been provided
    if plotTempContours and temp_levels==None:
        raise ValueError("Plot temperature contours was set to true but no contour values were provided.  Please set temp_levels to a list of temperaure values at which contours should be plotted")
    
    # Viscosity
    im = axs[0].pcolor(X, Y, np.log10(grid.eta_n),vmin=12, vmax=20)
    fig.colorbar(im, ax=axs[0],pad=0.0)                 # display colorbar
    axs[0].set(ylabel='y (m)', xlim=xlims, ylim=ylims)                          # label the y-axis (shared axis for x)
    axs[0].set_title('Viscosity log10(Pa s)')           # set plot title
    
    # Add temperature contours
    if plotTempContours:
        cs = axs[0].contour(X, Y, grid.T-273, levels=temp_levels, colors='w', linewidths=0.8)
        axs[0].clabel(cs, inline=True, fontsize=8, fmt='%d C')


    # get the mapping of markers to pixel positions
    marker_map = getMarkerPixelGrid(params, markers, grid, 401)
    box_size = [xlims[0], xlims[1], ylims[0],ylims[1]]
    
    # get the specific fields we want here
    mark_sigmaxx = getMarkerField(marker_map, markers.sigmaxx)
    mark_sigmaxy = getMarkerField(marker_map, markers.sigmaxy)
    mark_sigmaii = np.sqrt(mark_sigmaxx**2 + mark_sigmaxy**2)

    ###########################################################################
    # plot the stress
    im = axs[1].imshow(mark_sigmaii, origin='upper', aspect='auto', extent=box_size, vmin=0, vmax=1.75e5)             
    fig.colorbar(im, ax=axs[1],pad=0.0, extend='both')
    axs[1].set_title('$\\sigma_{ii}$ (Pa)')
    axs[1].set(ylabel='y (m)', xlim=xlims, ylim=ylims)
    
    # Add temperature contours
    if plotTempContours:
        cs = axs[1].contour(X, Y, grid.T-273, levels=temp_levels, colors='w', linewidths=0.8)
        axs[1].clabel(cs, inline=True, fontsize=8, fmt='%d C')

    # Plot accumulated strain
    mark_gii = getMarkerField(marker_map, markers.gII)
    im = axs[2].imshow(np.log10(mark_gii), origin='upper', aspect='auto', extent=box_size, vmin=-5, vmax=1.5)
    
    fig.colorbar(im, ax=axs[2],pad=0.0, extend='both')
    axs[2].set_title('Total strain (log10)')
    axs[2].set(xlabel='x (m)', ylabel = 'y (m)', xlim=xlims, ylim=ylims) 

    # Add temperature contours
    if plotTempContours:
        cs = axs[2].contour(X, Y, grid.T-273, levels=temp_levels, colors='w', linewidths=0.8)
        axs[2].clabel(cs, inline=True, fontsize=8, fmt='%d C')

    fig.suptitle('Time: %.3f Myr'%(t_curr*1e-6/(365.25*24*3600)))
    fig.savefig('%s/%s/allthree_%i.png'%(params.output_path, params.output_name, ntstp))




def animateAVar(grid_list, vxb_list, vyb_list, params, t_list, filename='animation.mp4'):
    """
    Animate a variable as a colormap with velocity arrows.
    """

    X, Y = np.meshgrid(grid_list[0].x, grid_list[0].y)

    fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
    im = ax.pcolor(X, Y, grid_list[0].rho, shading='nearest', vmin=0, vmax=3000)
    quiv = ax.quiver(grid_list[0].x, grid_list[0].y, vxb_list[0], np.flip(-vyb_list[0], 0))
    ax.set_title('Density')
    ax.set(ylabel='y (km)')
    ax.invert_yaxis()

    def update(frame):
        grid = grid_list[frame]
        vxb = vxb_list[frame]
        vyb = vyb_list[frame]
        im.set_array(grid.rho.ravel())
        quiv.set_UVC(vxb, np.flip(-vyb, 0))
        ax.set_title(f'Density - Time: {t_list[frame]:.3f} yr')
        return im, quiv

    anim = FuncAnimation(fig, update, frames=len(grid_list), blit=False)
    # anim.save(filename, writer='pillow')
    anim.save('%s/%s/%s'%(params.output_path, params.output_name, filename), writer='pillow')
    plt.close(fig)

    print('Animation made')