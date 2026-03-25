#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualisation routines for plotting code output

"""
import matplotlib.pyplot as plt
from matplotlib import figure
import numpy as np
from numba import jit

@jit
def basicGridVelocities(gridvx, gridvy, xnum, ynum):
    '''
    Interpolates the velocity values to the basic nodes, for visualisation only.

    Parameters
    ----------
    gridvx : ARRAY
        x-velocities at the staggered computation nodes.
    gridvy : ARRAY
        y-velocities at the staggered computation nodes.
    xnum : INT
        x-resolution of the simulation domain.
    ynum : INT
        y-resolution of the simulation domain.

    Returns
    -------
    vxb : ARRAY
        x-velocities at the basic nodes.
    vyb : ARRAY
        y-velocities at the basic nodes.

    '''
    vxb = np.zeros((ynum, xnum))
    vyb = np.zeros((ynum, xnum))
    
    for i in range(0,ynum):
        for j in range(0,xnum):
            vxb[i,j] = (gridvx[i,j] + gridvx[i+1,j])/2
            vyb[i,j] = (gridvy[i,j] + gridvy[i,j+1])/2
    
    return vxb, vyb


def plotAVar(grid, vxb, vyb, L_x, L_y, ntstp, t_curr):
    '''
    Plot a single variable as a colormap.

    Parameters
    ----------
    grid : Grid object
        Grid containing all simulation variables.
    vxb : ARRAY
        x velocities interpolated to the basic nodes.
    vyb : ARRAY
        y-velocities interpolated to the basic nodes.
    L_x : ARRAY
        Physical x-size of the simulation domain.
    L_y : ARRAY
        Physical y-size of the simulation domain.
    ntstp : INT
        Current timestep number.
    t_curr : FLOAT
        Current time (s).

    Returns
    -------
    None.

    '''
    
    X, Y = np.meshgrid(grid.x, grid.y)
    
    # create figure
    fig = figure.Figure(figsize=(9,3), constrained_layout=True)
    axs = fig.subplots(1,1, sharex=True, sharey=True)

    # plot the temperature as colormap
    im = axs.pcolor(X, Y, grid.T-273, shading='nearest', vmin=0, vmax=1400)
    fig.colorbar(im, ax=axs,pad=0.0)                                        # display colorbar
    axs.set_title('Temperature (C)')                                        # set plot title
    axs.set(ylabel='y (m)', xlabel='x (m)', xlim=(0e3, 550e3), ylim=(0, 300e3))                               # label the y-axis and x-axis
    axs.invert_yaxis()                                                      # plot increasing depth downward! 

    fig.suptitle('Time: %.3f Myr'%(t_curr*1e-6/(365.25*24*3600)))
    fig.savefig('./Figures/temp_%i.png'%(ntstp))


def plotSeveralVars(grid, vxb, vyb, L_x, L_y, ntstp, t_curr):
    '''
    Plot several different grid variables as colourmaps, default is 3x2 grid of plots.

    Parameters
    ----------
    grid : Grid object
        Grid containing all simulation variables.
    vxb : ARRAY
        x velocities interpolated to the basic nodes.
    vyb : ARRAY
        y-velocities interpolated to the basic nodes.
    L_x : FLOAT
        Physical x-size of the simulation domain.
    L_y : FLOAT
        Physical y-size of the simulation domain.
    ntstp : INT
        Current timestep number.
    t_curr : FLOAT
        Current time (s).

    Returns
    -------
    None.

    '''
    
    xres = grid.xnum
    yres = grid.ynum
    
    xres = grid.xnum
    yres = grid.ynum
    
    X, Y = np.meshgrid(grid.x, grid.y)
    
    # temperatures for countours
    temp_levels = [100, 150, 350, 450, 1300]
    
    # create figure and subplots
    fig = figure.Figure(figsize=(18,18), constrained_layout=True)
    axs = fig.subplots(3,1, sharex=True, sharey=True)

    # plot the density as colormap
    im = axs[0].pcolor(X, Y, grid.rho, shading='nearest', vmin=2200, vmax=3500)
    fig.colorbar(im, ax=axs[0],pad=0.0)        # display colorbar
    axs[0].set_title('Density (kg/m3) ')       # set plot title
    axs[0].set(ylabel='y (m)', xlim=(0e3, 550e3), ylim=(300e3, 0))                 # label the y-axis (shared axis for x)
    
    # add velocity arrows, not at every cell, step sets the spacing
    step = 5
    qu = axs[0].quiver(X[::step, ::step], Y[::step, ::step], vxb[:yres,:][::step, ::step], np.flip(-vyb[:,:xres],0)[::step, ::step])	    
    
    # Add temperature contours
    cs = axs[0].contour(X, Y, grid.T-273, levels=temp_levels, colors='w', linewidths=0.8)
    axs[0].clabel(cs, inline=True, fontsize=8, fmt='%d C')
    
    # plot increasing depth downward! (only needed for the annotations)
    axs[0].invert_yaxis()

    # Viscosity
    im = axs[1].pcolor(X, Y, np.log10(grid.eta_n),vmin=18, vmax=28)
    fig.colorbar(im, ax=axs[1],pad=0.0)                 # display colorbar
    axs[1].set(ylabel='y (m)', xlim=(0e3, 550e3), ylim=(300e3, 0))                          # label the y-axis (shared axis for x)
    axs[1].set_title('Viscosity log10(Pa s)')           # set plot title
    
    # Add temperature contours
    cs = axs[1].contour(X, Y, grid.T-273, levels=temp_levels, colors='w', linewidths=0.8)
    axs[1].clabel(cs, inline=True, fontsize=8, fmt='%d C')

	# Pressure
    im = axs[2].pcolor(X, Y, grid.P, shading='flat',vmin=0.1e9,vmax=9e9)
    fig.colorbar(im, ax=axs[2],pad=0.0)                 # display colorbar
    axs[2].set(ylabel='y (m)', xlabel='x (m)', xlim=(0e3, 550e3), ylim=(300e3, 0))          # label the x and y-axis
    axs[2].set_title('Pressure (Pa)')                   # set plot title
    
    # Add temperature contours
    cs = axs[2].contour(X, Y, grid.T-273, levels=temp_levels, colors='w', linewidths=0.8)
    axs[2].clabel(cs, inline=True, fontsize=8, fmt='%d C')


    fig.suptitle('Time: %.3f Myr'%(t_curr*1e-6/(365.25*24*3600)))
    
    fig.savefig('./Figures/densT_%i.png'%(ntstp))

def plotMarkerFields_Lithology(params, markers, grid, ntstp, t_curr):
    '''
    Plot the lithology and accumulated strain recorded by the markers.

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
    # first calculate a grid of interpolated marker values
    mark_com, mark_gii, mark_sigmaxx, mark_epsxx, mark_epsxy, mark_epsii, mark_sigmaii, mark_sigmaxy = get_marker_fields_vis(params.xsize, params.ysize, markers, grid)

    # create figure
    fig = figure.Figure(figsize=(9,3), constrained_layout=True)
    axs = fig.subplots(1,1, sharex=True, sharey=True)
    
    # create image grid and temperature contour levels
    X, Y = np.meshgrid(grid.x, grid.y)
    temp_levels = [100, 150, 350, 450, 1300]

    # plot the lithology as colormap
    im = axs.imshow(mark_com, origin='upper', aspect='auto', extent=[0,params.xsize,params.ysize,0])
    fig.colorbar(im, ax=axs,pad=0.0)                                                 # display colorbar
    axs.set_title('Lithology')                                                       # set plot title
    axs.set(ylabel='y (m)', xlabel ='x (m)', xlim=(0e3, 550e3), ylim=(300e3, 0))     # labels, limits                        
    
    fig.suptitle('Time: %.3f Myr'%(t_curr*1e-6/(365.25*24*3600)))
    fig.savefig(f'./figures/{params.output_name}/litho_%i.png'%(ntstp))


def plotMarkerFields(xsize, ysize, markers, grid, ntstp, t_curr):
    '''
    Plot the lithology and accumulated strain recorded by the markers.

    Parameters
    ----------
    xsize : FLOAT
        Physical x-size of the simulation domain.
    ysize : FLOAT
        Physical y-size of the simulation domain.
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
    # first calculate a grid of interpolated marker values
    mark_com, mark_gii, mark_sigmaxx, mark_epsxx, mark_epsxy, mark_epsii, mark_sigmaii, mark_sigmaxy = get_marker_fields_vis(xsize, ysize, markers, grid)
    
    # create figure, subplots
    fig = figure.Figure(figsize=(18,18), constrained_layout=True)
    axs = fig.subplots(3,1, sharex=True)
    
    # create image grid and temperature contour levels
    X, Y = np.meshgrid(grid.x, grid.y)
    temp_levels = [100, 150, 350, 450, 1300]

    # plot the stress
    im = axs[0].imshow(mark_sigmaii, origin='upper', aspect='auto', extent=[0, xsize, ysize, 0])             
    fig.colorbar(im, ax=axs[0],pad=0.0)                                                 # display colorbar
    axs[0].set_title('$\\sigma_{ii}$ (Pa)')                                                       # set plot title
    axs[0].set(ylabel='y (m)', xlim=(0e3, 550e3), ylim=(300e3, 0))                      # labels, limits
    
    # add temperature contours
    cs = axs[0].contour(X, Y, grid.T-273, levels=temp_levels, colors='w', linewidths=0.8)
    axs[0].clabel(cs, inline=True, fontsize=8, fmt='%d C')                     
    
    # plot normal stress components
    im = axs[1].imshow(mark_sigmaxx, origin='upper', aspect='auto', extent=[0, xsize, ysize, 0])                     # extent=[0,xsize,ysize,0]
    fig.colorbar(im, ax=axs[1],pad=0.0)                                                 # display colorbar
    axs[1].set_title('$\\sigma_{xx}$ (Pa)')                                                          # set plot title
    axs[1].set(ylabel = 'y (m)', xlim=(0e3, 550e3), ylim=(300e3, 0))                   # labels, limits
    
    # add temperature contours
    cs = axs[1].contour(X, Y, grid.T-273, levels=temp_levels, colors='w', linewidths=0.8)
    axs[1].clabel(cs, inline=True, fontsize=8, fmt='%d C')
    
    # plot shear stress components
    im = axs[2].imshow(mark_sigmaxy, origin='upper', aspect='auto', extent=[0, xsize, ysize, 0])                     # extent=[0,xsize,ysize,0]
    fig.colorbar(im, ax=axs[2],pad=0.0)                                                 # display colorbar
    axs[2].set_title('$\\sigma_{xy}$ (Pa)')                                              # set plot title
    axs[2].set(xlabel='x (m)', ylabel = 'y (m)', xlim=(0e3, 550e3), ylim=(300e3, 0))   # labels, limits
    
    # add temperature contours
    cs = axs[2].contour(X, Y, grid.T-273, levels=temp_levels, colors='w', linewidths=0.8)
    axs[2].clabel(cs, inline=True, fontsize=8, fmt='%d C')

    fig.suptitle('Time: %.3f Myr'%(t_curr*1e-6/(365.25*24*3600)))
    fig.savefig('./Figures/stress_%i.png'%(ntstp))
    
def plotMarkerFields2(xsize, ysize, markers, grid, ntstp, t_curr):
    '''
    Plot the lithology and accumulated strain recorded by the markers.

    Parameters
    ----------
    xsize : FLOAT
        Physical x-size of the simulation domain.
    ysize : FLOAT
        Physical y-size of the simulation domain.
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
    # first calculate a grid of interpolated marker values
    mark_com, mark_gii, mark_sigmaxx, mark_epsxx, mark_epsxy, mark_epsii, mark_sigmaii, mark_sigmaxy = get_marker_fields_vis(xsize, ysize, markers, grid)
    
    # create figure, subplots
    fig = figure.Figure(figsize=(18,18), constrained_layout=True)
    axs = fig.subplots(4,1, sharex=True)
    
    # create image grid and temperature contour levels
    X, Y = np.meshgrid(grid.x, grid.y)
    temp_levels = [100, 150, 350, 450, 1300]

    # plot the normal strain rate components
    im = axs[0].imshow(mark_epsxx, origin='upper', aspect='auto', extent=[0,xsize,ysize,0], vmin=-4e-14, vmax=4e-14)
    fig.colorbar(im, ax=axs[0],pad=0.0)                                                 # display colorbar
    axs[0].set_title('$\\dot\\epsilon_{xx}$ (1/s)')                                                       # set plot title
    axs[0].set(ylabel='y (m)', xlim=(0e3, 550e3), ylim=(300e3, 0))                      # labels, limits
    
    #add temperature contours
    cs = axs[0].contour(X, Y, grid.T-273, levels=temp_levels, colors='w', linewidths=0.8)
    axs[0].clabel(cs, inline=True, fontsize=8, fmt='%d C')                     
    
    # plot the shear strain rate components
    im = axs[1].imshow(mark_epsxy, origin='upper', aspect='auto', extent=[0,xsize,ysize,0], vmin=-4e-14, vmax=4e-14)
    fig.colorbar(im, ax=axs[1],pad=0.0)                                                 # display colorbar
    axs[1].set_title('$\\dot\\epsilon_{xy}$ (1/s)')                                                          # set plot title
    axs[1].set(ylabel = 'y (m)', xlim=(0e3, 550e3), ylim=(300e3, 0))                   # labels, limits
    
    # add temperature contours
    cs = axs[1].contour(X, Y, grid.T-273, levels=temp_levels, colors='w', linewidths=0.8)
    axs[1].clabel(cs, inline=True, fontsize=8, fmt='%d C')
    
    # plot normal stress components
    im = axs[2].imshow(mark_epsii, origin='upper', aspect='auto', extent=[0,xsize,ysize,0], vmin=-4e-14, vmax=4e-14)
    fig.colorbar(im, ax=axs[2],pad=0.0)                                                 # display colorbar
    axs[2].set_title('$\\dot \\epsilon_{ii}$ (1/s)')					# set plot title
    axs[2].set(ylabel='y (m)', xlim=(0e3, 550e3), ylim=(300e3, 0))                         # labels, limits
   
    # add temperature contours
    cs = axs[2].contour(X, Y, grid.T-273, levels=temp_levels, colors='w', linewidths=0.8)
    axs[2].clabel(cs, inline=True, fontsize=8, fmt='%d C')

    # Plot accumulated strain
    im = axs[3].imshow(np.log10(mark_gii), origin='upper', aspect='auto', extent=[0,xsize,ysize,0], vmin=-2, vmax=2)
    fig.colorbar(im, ax=axs[3],pad=0.0)                                                 # display colorbar
    axs[3].set_title('Total strain (log10)')                                              # set plot title
    axs[3].set(xlabel='x (m)', ylabel = 'y (m)', xlim=(0e3, 500e3), ylim=(300e3, 0)) 

    # add temperature contours
    cs = axs[3].contour(X, Y, grid.T-273, levels=temp_levels, colors='w', linewidths=0.8)
    axs[3].clabel(cs, inline=True, fontsize=8, fmt='%d C')

    fig.suptitle('Time: %.3f Myr'%(t_curr*1e-6/(365.25*24*3600)))
    fig.savefig('./Figures/strain_%i.png'%(ntstp))



@jit(nopython=True)
def get_marker_fields_vis(xsize, ysize, markers, grid):
    '''
    interpolates the marker lithology and strain rate onto a pixel grid for imaging.

    Parameters
    ----------
    xsize : FLOAT
        Physical x-size of the simulation domain.
    ysize : FLOAT
        Physical y-size of the simulation domain.
    markers : Markers object
        Contains all the marker values for each variable.
    grid : Grid object
        Contains all the grid variables at the current time.

    Returns
    -------
    mark_com : ARRAY
        Pixel grid of values of the ID (lithology) interpolated from markers.
    mark_gii : ARRAY
        Pixel grid of values of the strain interpolated from markers.

    '''
    
    # define the image resolution - want it to be about 100s in each dimension
    # but proportional to the grid size in each direction
    
    xres = 401
    yres = int(ysize/xsize*xres) + 1
    
    # xres = int(xsize/1000) + 1
    # yres = int(ysize/1000) + 1
    
    ngrid = 2
    
    sxstp = xsize/(xres - 1)
    systp = ysize/(yres - 1)
    
    # create marker visualization arrays
    mark_com = np.ones((yres, xres))*np.nan
    mark_dis = np.ones((yres, xres))*1e20
    mark_gii = np.ones((yres, xres))*np.nan 
    mark_sigmaxx = np.ones((yres, xres))*np.nan
    mark_epsxx = np.ones((yres, xres))*np.nan
    mark_epsxy = np.ones((yres, xres))*np.nan    
    mark_epsii = np.ones((yres, xres))*np.nan
    mark_sigmaxy = np.ones((yres, xres))*np.nan
    mark_sigmaii = np.ones((yres, xres))*np.nan

    # loop through markers
    for m in range(0,markers.num):
        
        # define pixel cell
        m1 = int((markers.x[m] - grid.x[0])/sxstp)
        m2 = int((markers.y[m] - grid.y[0])/systp)
        
        if (m1<0):
            m1 = 0
        elif (m1>xres-2):
            m1 = xres-2
 
        if (m2<0):
            m2 = 0
        elif (m2>yres-2):
            m2 = yres-2
        
        # define surrounding indicies
        m1min = m1-ngrid
        if (m1min<0):
            m1min = 0
        
        m1max = m1 + ngrid + 1
        if (m1max>xres-1):
            m1max = xres - 1
        
        m2min = m2-ngrid
        if (m2min < 0):
            m2min = 0
        
        m2max = m2 + ngrid + 1
        if (m2max>yres-1):
            m2max = yres-1
        
        # update pixels around marker
        for m10 in range(m1min, m1max):
            for m20 in range(m2min, m2max):
                # check distance to current cell
                dx = (markers.x[m] - grid.x[0]) - m10*sxstp
                dy = (markers.y[m] - grid.y[0]) - m20*systp

                dd = np.sqrt(dx**2 + dy**2)
                
                if (dd<mark_dis[m20, m10]):
                    mark_com[m20, m10] = markers.id[m]
                    mark_gii[m20, m10] = markers.gII[m]
                    mark_dis[m20, m10] = dd
                    mark_sigmaxx[m20, m10] = markers.sigmaxx[m]
                    mark_sigmaxy[m20, m10] = markers.sigmaxy[m]
                    mark_epsxx[m20, m10] = markers.epsxx[m]
                    mark_epsxy[m20, m10] = markers.epsxy[m]
                    mark_epsii[m20, m10] = np.sqrt(markers.epsxy[m]**2+markers.epsxy[m]**2)
                    mark_sigmaii[m20, m10] = np.sqrt(markers.sigmaxx[m]**2+markers.sigmaxy[m]**2)

    return mark_com, mark_gii, mark_sigmaxx, mark_epsxx, mark_epsxy, mark_epsii, mark_sigmaii, mark_sigmaxy



def makePlots(grid, vxb, vyb, params, ntstp, t_curr):
    """
    Wrapper function which calls all plotting routines, to simplify calling in 
    the run script.

    Parameters
    ----------
    grid : grid Object
        grid object containing the all the simulation variables on the grid.
    vxb : ARRAY
        x-component of velocities interpolated to basic grid nodes.
    vyb : ARRAY
        y-component of velocities interpolated to basic grid nodes.
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
    
    plotAVar(grid, vxb, vyb, params, ntstp, t_curr)
    plotSeveralVars(grid, vxb, vyb, params, ntstp, t_curr)
    

    
    
    
