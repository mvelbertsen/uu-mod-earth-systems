#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualisation routines

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
    Plot a single variable as a colormap, with velocity arrows annotated.

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
    
    xres = grid.xnum
    yres = grid.ynum
    
    X, Y = np.meshgrid(grid.x, grid.y)
    
    fig = figure.Figure(figsize=(9,9), constrained_layout=True)
    
    axs = fig.subplots(1,1, sharex=True, sharey=True)

    # plot the density as colormap
    im = axs.pcolor(X, Y, grid.rho, shading='nearest', vmin=1000, vmax=3300)
    fig.colorbar(im, ax=axs,pad=0.0) # display colorbar
    axs.set_title('Density')     # set plot title
    axs.set(ylabel='y (km)')         # label the y-axis (shared axis for x)
    qu = axs.quiver(grid.x, grid.y, vxb[:yres,:], np.flip(-vyb[:,:xres],0)) 
    axs.invert_yaxis()

    fig.suptitle('Time: %.3f Myr'%(t_curr*1e-6/(365.25*24*3600)))
    fig.savefig('./Figures/rho_tstp_%i.png'%(ntstp))
    #plt.close()

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
    XP, YP = np.meshgrid(grid.cx, grid.cy)
    
    fig = figure.Figure(figsize=(18,12), constrained_layout=True)
    
    axs = fig.subplots(2,3, sharex=True, sharey=True)

    # plot the density as colormap
    im = axs[0,0].pcolor(X, Y, grid.rho, shading='nearest', vmin=1000, vmax=3300)
    fig.colorbar(im, ax=axs[0,0],pad=0.0) # display colorbar
    axs[0,0].set_title('Density')     # set plot title
    axs[0,0].set(ylabel='y (km)')         # label the y-axis (shared axis for x)
    
    qu = axs[0,0].quiver(grid.x, grid.y, vxb[:yres,:], np.flip(-vyb[:,:xres],0)) 
    axs[0,0].invert_yaxis()

    # plot the pressure as colormap
    im = axs[0,1].pcolor(X, Y, grid.P, shading='flat',vmin=0.1e9,vmax=9e9)
    fig.colorbar(im, ax=axs[0,1],pad=0.0) # display colorbar
    axs[0,1].set_title('Pressure')     # set plot title

    # plot vx as colormap
    im = axs[1,0].pcolor(X, Y, vxb, shading='nearest',vmin=-5e-10, vmax=5e-10)
    fig.colorbar(im, ax=axs[1,0],pad=0.0) # display colorbar
    axs[1,0].set_title('Vx')     # set plot title
    axs[1,0].set(ylabel='y (km)', xlabel='x (km)')         # label the y-axis (shared axis for x)

    # plot vy as colormap
    im = axs[1,1].pcolor(X, Y, vyb, shading='nearest',vmin=-5e-10, vmax=5e-10)
    fig.colorbar(im, ax=axs[1,1],pad=0.0) # display colorbar
    axs[1,1].set_title('Vy')     # set plot title
    axs[1,1].set(xlabel='x (km)')         
    
    # T
    im = axs[0,2].pcolor(X, Y, grid.T, shading='nearest',vmin=273, vmax=1800)
    fig.colorbar(im, ax=axs[0,2],pad=0.0) # display colorbar
    axs[0,2].set_title('Temperature')     # set plot title

    # sigxy
    im = axs[1,2].pcolor(X, Y, np.log10(grid.eta_s),vmin=18, vmax=25)
    fig.colorbar(im, ax=axs[1,2],pad=0.0) # display colorbar
    axs[1,2].set_title('Viscosity')     # set plot title
    
    fig.suptitle('Time: %.3f Myr'%(t_curr*1e-6/(365.25*24*3600)))

    fig.savefig('./Figures/densT_tstp_%i.png'%(ntstp))
    #plt.close()


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
    
    mark_com, mark_gii = get_marker_fields_vis(xsize, ysize, markers, grid)
    
    fig = figure.Figure(figsize=(12,18), constrained_layout=True)
    
    axs = fig.subplots(2,1, sharex=True)

    # plot the density as colormap
    im = axs[0].imshow(mark_com, origin='upper', aspect='auto', extent=[0,xsize,ysize,0])
    fig.colorbar(im, ax=axs[0],pad=0.0) # display colorbar
    axs[0].set_title('Lithology')     # set plot title
    axs[0].set(ylabel='y (m)', xlim=(50e3,350e3), ylim=(80e3,0))
    
    im = axs[1].imshow(np.log10(mark_gii), origin='upper', aspect='auto', extent=[0,xsize,ysize,0], vmin=-2, vmax=2)
    fig.colorbar(im, ax=axs[1],pad=0.0) # display colorbar
    axs[1].set_title('Strain')     # set plot title
    axs[1].set(xlabel='x (m)', ylabel='y (m)', xlim=(50e3,350e3), ylim=(150e3,0))
    
    fig.suptitle('Time: %.3f Myr'%(t_curr*1e-6/(365.25*24*3600)))
    
    fig.savefig('./Figures/lithology_tstp_%i.png'%(ntstp))
    #plt.close()
    

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
    
    # define the image resolution
    xres = int(xsize/1000) + 1
    yres = int(ysize/1000) + 1
    
    ngrid = 2
    
    sxstp = xsize/(xres - 1)
    systp = ysize/(yres - 1)
    
    # create marker visualization arrays
    mark_com = np.ones((yres, xres))*np.nan
    mark_dis = np.ones((yres, xres))*1e20
    mark_gii = np.ones((yres, xres))*np.nan
    
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

    return mark_com, mark_gii




    
    
    