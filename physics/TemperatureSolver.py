#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temperature equation solver.  

Uses scipy's sparse arrays and sparse direct matrix solver to solve the temperature equation in 2D.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import bsr_matrix, csr_matrix, coo_array
from scipy.sparse.linalg import spsolve
from numba import jit


@jit(nopython=True)
def constructTempRHS(grid, params):
    '''
    Computes the (spatial domain) RHS values for the temperature eqn.

    Parameters
    ----------
    grid : Grid
        Grid object containing all the grid variables.
    params : Parameters
        Parameters object containing all the simulation parameters.

    Returns
    -------
    RT : ARRAY
        Calculated RHS of temperature eqn.

    '''
    
    # first include the radiogenic part defined by lithology
    RT = np.zeros((grid.ynum, grid.xnum))
    RT += grid.H_r
    
    # add in adiabatic and/or shear heating terms
    for i in range(1, grid.ynum-1):
        for j in range(1, grid.xnum-1):
            
            if (params.adia_yn == 1):
                # apply adiabatic heating
                RT[i,j] += grid.H_a[i,j]*grid.rho[i,j]*(params.gx*(grid.vx[i,j] + grid.vx[i+1,j])\
                                                      + params.gy*(grid.vy[i,j] + grid.vy[i,j+1]))/2
            
            if (params.frict_yn == 1):
                # apply shear heating
                RT[i,j] += grid.sigxy2[i,j]**2/grid.eta_s[i,j]
                RT[i,j] += ( grid.sigxx2[i-1,j-1]**2/grid.eta_n[i-1,j-1] + grid.sigxx2[i-1,j]**2/grid.eta_n[i-1,j]\
                           + grid.sigxx2[i,j-1]**2/grid.eta_n[i,j-1] + grid.sigxx2[i,j]**2/grid.eta_n[i,j])/4
                
    return RT


@jit(nopython=True)
def S_to_grid(S,xres, yres):
    '''
    Transforms the solution vector of the linear system back to the spatial grid

    Parameters
    ----------
    S : ARRAY
        Solution vector from the linear system.
    xres : INT
        Number of nodes in the x-direction.
    yres : INT
        Number of nodes in the y-direction.

    Returns
    -------
    T : ARRAY
        2D spatial array of the solution of the linear system.

    '''
    T = np.zeros((yres, xres))
    for j in range(0,xres):
        for i in range(0,yres):
            # define the global indicies k
            k = j*(yres) + i
            
            # revert solution
            T[i,j] = S[k]
    return T

@jit(nopython=True)
def TemperatureConstructMatrix(xnum, ynum, gridx, gridy, kt, rho_Cp, tstep, B_top, B_bottom, B_left, B_right, R_heat, T_k):
    '''
    Constructs the matrix system to be solved for the temperature eqn.

    Parameters
    ----------
    xnum : INT
        Number of nodes in x-direction.
    ynum : INT
        Number of nodes in y-direction.
    gridx : ARRAY
        The x positions of the T nodes.
    gridy : ARRAY
        The y positions of the T nodes.
    kt : ARRAY
        Thermal conductivity at the T nodes.
    rho_Cp : ARRAY
        Density * C_P at the T nodes.
    tstep : FLOAT
        Current timestep.
    B_top : ARRAY
        Top temperature BCs.  Array has 2 columns, values in each are defined as:
        T[i,j] = B_top[0] + B_top[1]*T[i+1,j]
    B_bottom : ARRAY
        Bottom temperature BCs.  Array has 2 columns, values in each are defined as:
        T[i,j] = B_bottom[0] + B_bottom[1]*T[i-1,j]
    B_left : ARRAY
        Left temperature BCs.  Array has 2 columns, values in each are defined as:
        T[i,j] = B_left[0] + B_left[1]*T[i,j+1]
    B_right : ARRAY
        Right temperature BCs.  Array has 2 columns, values in each are defined as:
        T[i,j] = B_right[0] + B_right[1]*T[i,j-1]
    R_heat : ARRAY
        Heating term at the T nodes.
    T_k : ARRAY
        Temperature (at the T nodes) from previous timestep.

    Returns
    -------
    rows : LIST
        row indicies of non-zero elements in the matrix
    cols : LIST
        column indicies of non-zero elements in the matrix
    data : LIST
        values in the matrix at the locations specified in rows, cols
    R : ARRAY
        RHS vector.
    xstp : ARRAY
        x-spacing between basic nodes.
    xstpc : ARRAY
        x-spacing between pressure nodes.
    ystp : ARRAY
        y-spacing between basic nodes.
    ystpc : ARRAY
        y-spacing between pressure nodes.

    '''
    
    # first compute grid steps for the basic nodes 
    xstp = gridx[1:] - gridx[:-1]
    ystp = gridy[1:] - gridy[:-1]

    # and the qx, qy nodes
    xstpc = (gridx[2:] - gridx[:-2])/2
    ystpc = (gridy[2:] - gridy[:-2])/2
    
    # set up the matricies L, R of the system to solve
    rows = []
    cols = []
    data = []
    R = np.zeros((xnum*ynum))
    
    # fill the arrays
    for i in range(0, ynum):
        for j in range(0, xnum):
            # set the global index
            k = i + ynum*j
            
            # boundary conditions
            if (i==0 or i==ynum-1 or j==0 or j==xnum-1):
                    
                # Upper BC
                if (i==0 and j>0 and j<xnum-1):
                    # LHS
                    rows.append(k)
                    cols.append(k)
                    data.append(1)
                    
                    rows.append(k)
                    cols.append(k+1)
                    data.append(-B_top[j,1])
                    # RHS
                    R[k] = B_top[j,0]
                    
                # Lower BC
                if (i==ynum-1 and j>0 and j<xnum-1):
                    # LHS
                    rows.append(k)
                    cols.append(k)
                    data.append(1)
                    
                    rows.append(k)
                    cols.append(k-1)
                    data.append(-B_bottom[j,1])
                    # RHS
                    R[k] = B_bottom[j,0]
                    
                # left BC
                if (j==0):
                    # LHS
                    rows.append(k)
                    cols.append(k)
                    data.append(1)
                    
                    rows.append(k)
                    cols.append(k+ynum)
                    data.append(-B_left[i,1])
                    # RHS
                    R[k] = B_left[i,0]
                    
                # right BC
                if (j==xnum-1):
                    # LHS
                    rows.append(k)
                    cols.append(k)
                    data.append(1)
                    
                    rows.append(k)
                    cols.append(k-ynum)
                    data.append(-B_right[i,1])
                    # RHS
                    R[k] = B_right[i,0]
                    
            # interior points
            else:         
                # RHS
                R[k] = R_heat[i,j] + T_k[i,j]*rho_Cp[i,j]/tstep
                
                # LHS
                # center node
                rows.append(k)
                cols.append(k)
                data.append(rho_Cp[i,j]/tstep + ( (kt[i,j-1] + kt[i,j])/xstp[j-1] + (kt[i,j] + kt[i,j+1])/xstp[j])/(2*xstpc[j-1])\
                                          + ( (kt[i-1,j] + kt[i,j])/ystp[i-1] + (kt[i,j] + kt[i+1,j])/ystp[i])/(2*ystpc[i-1]))
                # left node
                rows.append(k)
                cols.append(k-ynum)
                data.append(-(kt[i,j-1] + kt[i,j])/(2*xstp[j-1]*xstpc[j-1]))
                # right node
                rows.append(k)
                cols.append(k+ynum)
                data.append(-(kt[i,j] + kt[i,j+1])/(2*xstp[j]*xstpc[j-1]))
                # upper node
                rows.append(k)
                cols.append(k-1)
                data.append(-(kt[i-1,j] + kt[i,j])/(2*ystp[i-1]*ystpc[i-1]))
                # lower node
                rows.append(k)
                cols.append(k+1)
                data.append(-(kt[i,j] + kt[i+1,j])/(2*ystp[i]*ystpc[i-1]))
    
    return rows, cols, data, R, xstp, xstpc, ystp, ystpc     


def TemperatureSolver(tstep, xnum, ynum, gridx, gridy, kt, rho_Cp, B_top, B_bottom, B_left, B_right, R_heat, T_k):
    '''
    Formulates and solves the heat conservation eqn on a 2D, irregularly spaced grid
    using scipy sparse matricies and solver.

    Parameters
    ----------
    tstep : FLOAT
        Current timestep.
    xnum : INT
        Number of nodes in x-direction.
    ynum : INT
        Number of nodes in y-direction.
    gridx : ARRAY
        The x positions of the T nodes.
    gridy : ARRAY
        The y positions of the T nodes.
    kt : ARRAY
        Thermal conductivity at the T nodes.
    rho_Cp : ARRAY
        Density * C_P at the T nodes.
    B_top : ARRAY
        Top temperature BCs.  Array has 2 columns, values in each are defined as:
        T[i,j] = BT_top[0] + BT_top[1]*T[i+1,j]
    B_bottom :  ARRAY
        Array specifying the boundary T for the bottom (y=L_y) of the domain.
    B_left :  ARRAY
        Array specifying the boundary T for the left (x=0) of the domain.
    B_right :  ARRAY
        Array specifying the boundary T for the right (x=L_x) of the domain.
    R_heat : ARRAY
        Heating term at the T nodes.
    T_k : ARRAY
        Temperature (at the T nodes) from previous timestep.

    Returns
    -------
    T_k_new : ARRAY
        2D array containing the new temperatures.
    T_res : ARRAY
        Calculated residuals.

    '''
    
    
    # construct the temperature matrix system to solve
    rows, cols, data, R, xstp, xstpc, ystp, ystpc = TemperatureConstructMatrix(xnum, ynum, gridx, gridy, kt, rho_Cp, tstep,\
                                                                B_top, B_bottom, B_left, B_right, R_heat, T_k)
                
    # now we can solve the system, first convert to sparse form
    L = coo_array((data, (rows, cols)), shape=(xnum*ynum, xnum*ynum)).tocsr()
    # using scipy's spsolve for sparse matricies
    S = spsolve(L,R)
    
    # reload solution to spatial grid (using fn defined above!)
    T_k_new = S_to_grid(S, xnum, ynum)

    # compute the residuals
    T_res = calculateResiduals(xnum, ynum, xstp, xstpc, ystp, ystpc, rho_Cp, T_k_new, T_k, kt, R_heat, tstep)
    
                
    return T_k_new, T_res

@jit
def calculateResiduals(xnum, ynum, xstp, xstpc, ystp, ystpc, rho_Cp, T_k_new, T_k, kt, R_heat, tstep):
    '''
    Calculates the residuals of the solver.

    Parameters
    ----------
    xnum : INT
        Number of normal nodes in x-direction.
    ynum : INT
        Number of normal nodes in y-direction.
    xstp : ARRAY
        Spacing between basic nodes in x-direction.
    xstpc : ARRAY
        Spacing between pressure nodes in x-direction.
    ystp : ARRAY
        Spacing between basic nodes in y-direction.
    ystpc : ARRAY
        Spacing between pressure nodes in y-direction.
    rho_Cp : ARRAY
        Density * C_P at the T nodes.
    T_k_new : ARRAY
        The newly calculated temperatures.
    T_k : ARRAY
        Temperature (at the T nodes) from previous timestep.
    kt : ARRAY
        Thermal conductivity at the T nodes.
    R_heat : ARRAY
        Heating term at the T nodes.
    tstep : FLOAT
        Current timestep.

    Returns
    -------
    T_res : ARRAY
        Residuals for the temperature eqn.

    '''
    
    # initialize array
    T_res = np.zeros((ynum, xnum))
    for i in range(0, ynum):
        for j in range(0, xnum):
            # boundary conditions, residuals are zero
            if (i==1 or j==1 or i==ynum-1 or j==xnum-1):
                T_res[i,j] = 0
            else:
                # compute current temperature eqn residual
                # Ht-DT/dt
                T_res[i,j] = R_heat[i,j] - rho_Cp[i,j]*(T_k_new[i,j] - T_k[i,j])/tstep
                # -dq/dx
                T_res[i,j] += ((kt[i,j] + kt[i,j+1])*(T_k_new[i,j+1] - T_k_new[i,j])/xstp[j]\
                              -(kt[i,j-1] + kt[i,j])*(T_k_new[i,j] - T_k_new[i,j-1])/xstp[j-1])/(2*xstpc[j-1])
                # -dq/dy
                T_res[i,j] += ((kt[i,j] + kt[i+1,j])*(T_k_new[i+1,j] - T_k_new[i,j])/ystp[i]\
                              -(kt[i-1,j] + kt[i,j])*(T_k_new[i,j] - T_k_new[i-1,j])/ystp[i-1])/(2*ystpc[i-1])
                    
    return T_res
