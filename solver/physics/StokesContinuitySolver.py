#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stokes and Continuity equations solver.  

Uses scipy's sparse arrays and sparse direct matrix solver to solve the Stokes and Continuity equations in 2D.

"""
import numpy as np
from scipy.sparse import bsr_matrix, csr_matrix, coo_array
from scipy.sparse.linalg import spsolve
from numba import jit


@jit(nopython=True)
def constructStokesRHS(grid, grid0, params, xnum, ynum):
    '''
    Computes the RHS (spatial domain) vectors for the vx, vy and continuity eqns.

    Parameters
    ----------
    grid : Grid
        Grid object containing all the grid variables.
    grid0 : Grid
        Grid object containing all the previous step's grid variables.
    params : Parameters
        Parameters object containing all the simulation parameters.
    xnum : INT
        x-resolution of the simulation domain.
    ynum : INT
        y-resolution of the simulation domain.

    Returns
    -------
    RX : ARRAY
        RHS values for the x-velocity Stoke's eqn.
    RY : ARRAY
        RHS values for the y-velocity Stoke's eqn.
    RC : ARRAY
        RHS values for the continuity eqn.

    '''
    
    # create arrays for storing RHS values in
    RX = np.zeros((ynum+1, xnum))
    RY = np.zeros((ynum, xnum+1))
    RC = np.zeros((ynum-1, xnum-1))
    
    for i in range(1,ynum):
        for j in range(1,xnum):
            # x-Stokes eqn
            if (j<xnum-1):
                RX[i,j] = -params.gx*(grid.rho[i,j]+grid.rho[i-1,j])/2
                RX[i,j] += -(grid0.sigxx[i-1,j] - grid0.sigxx[i-1,j-1])/grid.xstpc[j]
                RX[i,j] += -(grid0.sigxy[i,j] - grid0.sigxy[i-1,j])/grid.ystp[i-1]
                
            if (i<ynum-1):
                RY[i,j] = -params.gy*(grid.rho[i,j]+grid.rho[i,j-1])/2
                #TODO: this is diff sign to x direction - why?
                RY[i,j] += (grid0.sigxx[i,j-1] - grid0.sigxx[i-1,j-1])/grid.ystpc[i]
                RY[i,j] += -(grid0.sigxy[i,j] - grid0.sigxy[i,j-1])/grid.xstp[j-1]
    
    return RX, RY, RC


@jit(nopython=True)
def S_to_grid_Stokes(S, xres, yres, Pscale, B_top, B_bottom, B_left, B_right):
    '''
    Translates the solution of the linear system back to spatial grids for each
    of vx, vy ,P.  Also enforces velocity BCs

    Parameters
    ----------
    S : Array
        Vector containing the solution of the linear system solved for the 
        Stokes and Continuty eqns.
    xres : INT
        Number of nodes in x-direction.
    yres : INT
        Number of nodes in y-direction.
    Pscale : FLOAT
        Pressure scaling constant.
    B_top : ARRAY
        Array (xres+1,4) containing the top BC for both x and y vector components.
    B_bottom : ARRAY
        Array (xres+1,4) containing the bottom BC for both x and y vector components.
    B_left : ARRAY
        Array (yres+1,4) containing the left BC for both x and y vector components.
    B_right : ARRAY
        Array (yres+1,4) containing the right BC for both x and y vector components.

    Returns
    -------
    vx : ARRAY
        The x velocity component.
    vy : ARRAY
        The y velocity component.
    P : ARRAY
        Pressure.

    '''
    # create spatial arrays
    vx = np.zeros((yres+1,xres))
    vy = np.zeros((yres,xres+1))
    P = np.zeros((yres-1,xres-1))
    
    for j in range(0,xres-1):
        for i in range(0,yres-1):
            # define the global indicies k
            kvx = (j*(yres-1) + i)*3
            kvy = kvx+1
            kP = kvx+2
            
            # revert solution
            vx[i+1,j+1] = S[kvx]
            vy[i+1,j+1] = S[kvy]
            P[i,j] = S[kP]*Pscale

    # apply BCs for velocities
    # vx, left right
    vx[:,0] = B_left[:,0] + B_left[:,1]*vx[:,1]
    vx[:,xres-1] = B_right[:,0] + B_right[:,1]*vx[:,xres-2]
    # vx, top bottom
    vx[0,:] = B_top[:xres,0] + B_top[:xres,1]*vx[1,:]
    vx[yres,:] = B_bottom[:xres,0] + B_bottom[:xres,1]*vx[yres-1,:]
    
    # vy left right
    vy[:,0] = B_left[:yres,2] + B_left[:yres,3]*vy[:,1]
    vy[:,xres] = B_right[:yres,2] + B_right[:yres,3]*vy[:,xres-1]
    # vy, top bottom
    vy[0,:] = B_top[:,2] + B_top[:,3]*vy[1,:]
    vy[yres-1,:] = B_bottom[:,2] + B_bottom[:,3]*vy[yres-2,:]
    
    return vx, vy, P

@jit(nopython=True)
def StokesConstructMatrix(xnum, ynum, xstp, xstpc, ystp, ystpc, xstp_av, ystp_av, Pscale, Pnorm, Bpres,\
                          eta_s, eta_n, R_x, R_y, R_C, B_top, B_bottom, B_left, B_right, B_intern):
    '''
    Constructs the matrix and RHS vector system for solving the Stoke's + continuity eqns.

    Parameters
    ----------
    xxnum : INT
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
    xstp_av : FLOAT
        Average node spacing in x-direction.
    ystp_av : FLOAT
        Average node spacing in y-direction.
    Pscale : FLOAT
        Pressure scaling value for numerical stability.
    Pnorm : FLOAT
        Pressure chosen for the top-left ghost zone/ or for top face as BC.
    Bpres : INT
        Pressure boundary condition mode.
    eta_s : ARRAY
        The shear stress viscosity values.
    eta_n : ARRAY
        The normal stress viscosity values.
    R_x : ARRAY
        RHS values of the x-component momentum equation.
    R_y : ARRAY
        RHS values of the y-component momentum equation
    R_C : ARRAY
        RHS values of the continuity equation
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
        B_intern[3] = prescribed x-velocity value.
        B_intern[4] = x-index of vy nodes with prescribed velocity (-1 is not in use)
        B_intern[5-6] = min/max y-index of the wall
        B_intern[7] = prescribed y-velocity value.

    Returns
    -------
    rows : LIST
        row indicies of non-zero elements in the matrix
    cols : LIST
        column indicies of non-zero elements in the matrix
    data : LIST
        values in the matrix at the locations specified in rows, cols
    R : ARRAY
        The vector of RHS values (xnum-1*ynum-1*3).

    '''
    
    # horizontal shift index
    ynum3 = 3*(ynum-1)
    
    N = (xnum-1)*(ynum-1)*3
    
    data = []
    rows = []
    cols = []
    
    R = np.zeros((N))
    
    for i in range(0,ynum-1):
       for j in range(0,xnum-1):
           # define the global indicies k
           kvx = (j*(ynum-1) + i)*3
           kvy = kvx+1
           kP = kvx+2
           
           ####################### x-Stokes eqn ###############################
           if (j<xnum-2 and (j!=B_intern[0] or i<B_intern[1] or i>B_intern[2])):
               # we are not at an internal fixed velocity boundary (or the right boundary),
               # apply internal stencils
               
               # Right part
               R[kvx] = R_x[i+1, j+1]
               
               # central Vx-node
               rows.append(kvx)
               cols.append(kvx)
               data.append(-2*(eta_n[i,j+1]/xstp[j+1] + eta_n[i,j]/xstp[j])/xstpc[j+1]\
                             - (eta_s[i+1,j+1]/ystpc[i+1] + eta_s[i,j+1]/ystpc[i])/ystp[i])
                             
               # left Vx node
               if (j>0):
                   # we're not at the left boundary
                   kvx_l = kvx-ynum3
                   rows.append(kvx)
                   cols.append(kvx_l)
                   data.append(2*eta_n[i,j]/xstp[j]/xstpc[j+1])
               else:
                   # we are at the boundary, use BCs
                   rows.append(kvx)
                   cols.append(kvx)
                   data.append(B_left[i+1,1]*2*eta_n[i,j]/xstp[j]/xstpc[j+1])
                   
                   R[kvx] -= B_left[i+1,0]*2*eta_n[i,j]/xstp[j]/xstpc[j+1]
               
               # Right vx node
               if (j < xnum-3):
                   kvx_r = kvx + ynum3
                   rows.append(kvx)
                   cols.append(kvx_r)
                   data.append(2*eta_n[i,j+1]/xstp[j+1]/xstpc[j+1])
               else:
                   # at the right boundary
                   rows.append(kvx)
                   cols.append(kvx)
                   data.append(B_right[i+1,1]*2*eta_n[i,j+1]/xstp[j+1]/xstpc[j+1])
                   R[kvx] -= B_right[i+1,0]*2*eta_n[i,j+1]/xstp[j+1]/xstpc[j+1]
                
               # top vx node
               if (i > 0):
                   kvx_t = kvx -3
                   rows.append(kvx)
                   cols.append(kvx_t)
                   data.append(eta_s[i,j+1]/ystpc[i]/ystp[i])
               else:
                   # at the top boundary
                   rows.append(kvx)
                   cols.append(kvx)
                   data.append(B_top[j+1,1]*eta_s[i,j+1]/ystpc[i]/ystp[i])
                   R[kvx] -= B_top[j+1,0]*eta_s[i,j+1]/ystpc[i]/ystp[i]
                   
               # bottom vx node
               if (i < ynum-2):
                   kvx_b = kvx + 3
                   rows.append(kvx)
                   cols.append(kvx_b)
                   data.append(eta_s[i+1,j+1]/ystpc[i+1]/ystp[i])
               else:
                   rows.append(kvx)
                   cols.append(kvx)
                   data.append(B_bottom[j+1,1]*eta_s[i+1, j+1]/ystpc[i+1]/ystp[i])
                   R[kvx] -= B_bottom[j+1,0]*eta_s[i+1,j+1]/ystpc[i+1]/ystp[i]
                   
               # vy 
               # top left vy node
               if (i > 0):
                   kvy_tl = kvx - 3 + 1
                   rows.append(kvx)
                   cols.append(kvy_tl)
                   data.append(eta_s[i,j+1]/xstpc[j+1]/ystp[i])
               else:
                   kvy_bl = kvx + 1
                   rows.append(kvx)
                   cols.append(kvy_bl)
                   data.append(B_top[j+1,3]*eta_s[i,j+1]/xstpc[j+1]/ystp[i])
                   R[kvx] -= B_top[j+1,2]*eta_s[i,j+1]/xstpc[j+1]/ystp[i]

               # top right vy node
               if (i > 0):
                   kvy_tr = kvx - 3 + 1 + ynum3
                   rows.append(kvx)
                   cols.append(kvy_tr)
                   data.append(-eta_s[i,j+1]/xstpc[j+1]/ystp[i])
               else:
                   kvy_br = kvx + 1 + ynum3
                   rows.append(kvx)
                   cols.append(kvy_br)
                   data.append(-B_top[j+2,3]*eta_s[i,j+1]/xstpc[j+1]/ystp[i])
                   R[kvx] += B_top[j+2,2]*eta_s[i,j+1]/xstpc[j+1]/ystp[i]
               
               # bottom-left vy node
               if (i < ynum-2):
                   kvy_bl = kvx + 1
                   if (i > 0):
                       rows.append(kvx)
                       cols.append(kvy_bl)
                       data.append(-eta_s[i+1,j+1]/xstpc[j+1]/ystp[i])
                   else:
                       rows.append(kvx)
                       cols.append(kvy_bl)
                       data.append(-eta_s[i+1,j+1]/xstpc[j+1]/ystp[i])
               else:
                   kvy_tl = kvx - 3 + 1
                   rows.append(kvx)
                   cols.append(kvy_tl)
                   data.append(-B_bottom[j+1,3]*eta_s[i+1,j+1]/xstpc[j+1]/ystp[i])
                   R[kvx] += B_bottom[j+1,2]*eta_s[i+1,j+1]/xstpc[j+1]/ystp[i]

               # bottom right vy node
               if (i < ynum-2):
                   kvy_br = kvx + 1 + ynum3
                   if (i > 0):
                       rows.append(kvx)
                       cols.append(kvy_br)
                       data.append(eta_s[i+1,j+1]/xstpc[j+1]/ystp[i])
                   else:
                       rows.append(kvx)
                       cols.append(kvy_br)
                       data.append(eta_s[i+1,j+1]/xstpc[j+1]/ystp[i])
               else:
                   kvy_tr = kvx - 3 + 1 + ynum3
                   rows.append(kvx)
                   cols.append(kvy_tr)
                   data.append(B_bottom[j+2,3]*eta_s[i+1,j+1]/xstpc[j+1]/ystp[i])
                   R[kvx] += - B_bottom[j+2,2]*eta_s[i+1,j+1]/xstpc[j+1]/ystp[i]
               
               # Pressure
               # Left P node
               kp_l=kvx+2
               rows.append(kvx)
               cols.append(kp_l)
               data.append(Pscale/xstpc[j+1])
               # Right P node
               kp_r = kvx + 2 + ynum3
               rows.append(kvx)
               cols.append(kp_r)
               data.append(-Pscale/xstpc[j+1])
               
           else:
               # at some kind of boundary
               rows.append(kvx)
               cols.append(kvx)
               data.append(2*Pscale/(xstp_av + ystp_av))
               if (j!=B_intern[0] or i<B_intern[1] or i>B_intern[2]):
                   # at the external boundary
                   R[kvx] = 0
               else:
                   # Internal prescribed horizontal velocity
                   R[kvx] = 2*Pscale/(xstp_av + ystp_av)*B_intern[3]
        
           
           ###################### y Stokes eqn ################################
           if (i<ynum-2 and (j!=B_intern[4] or i<B_intern[5] or i>B_intern[6])):
               # we are not at an internal fixed velocity boundary (or the right boundary),
               # apply internal stencils
               
               # Right part
               R[kvy] = R_y[i+1, j+1]
               
               # central Vx-node
               rows.append(kvy)
               cols.append(kvy)
               data.append(-2*(eta_n[i+1,j]/ystp[i+1] + eta_n[i,j]/ystp[i])/ystpc[i+1]\
                             - (eta_s[i+1,j+1]/xstpc[j+1] + eta_s[i+1,j]/xstpc[j])/xstp[j])
                             
               # top vy node
               if (i > 0):
                   # we're not at the left boundary
                   kvy_t = kvy - 3
                   rows.append(kvy)
                   cols.append(kvy_t)
                   data.append(2*eta_n[i,j]/ystp[i]/ystpc[i+1])
               else:
                   # we are at the boundary, use BCs
                   rows.append(kvy)
                   cols.append(kvy)
                   data.append(B_top[j+1,3]*2*eta_n[i,j]/ystp[i]/ystpc[i+1])
                   R[kvy] -= B_top[j+1,2]*2*eta_n[i,j]/ystp[i]/ystpc[i+1]
                   
               # bottom vy node
               if (i < ynum-3):
                   kvy_b = kvy + 3
                   rows.append(kvy)
                   cols.append(kvy_b)
                   data.append(2*eta_n[i+1,j]/ystp[i+1]/ystpc[i+1])
               else:
                   # at the right boundary
                   rows.append(kvy)
                   cols.append(kvy)
                   data.append(B_bottom[j+1,3]*2*eta_n[i+1,j]/ystp[i+1]/ystpc[i+1])
                   R[kvy] -= B_bottom[j+1,2]*2*eta_n[i+1,j]/ystp[i+1]/ystpc[i+1]
                
               # left vy node
               if (j > 0):
                   kvy_l = kvy - ynum3
                   rows.append(kvy)
                   cols.append(kvy_l)
                   data.append(eta_s[i+1,j]/xstpc[j]/xstp[j])
               else:
                   # at the left boundary
                   rows.append(kvy)
                   cols.append(kvy)
                   data.append(B_left[i+1,3]*eta_s[i+1,j]/xstpc[j]/xstp[j])
                   R[kvy] -= B_left[i+1,2]*eta_s[i+1,j]/xstpc[j]/xstp[j]
                   
               # right vy node
               if (j < xnum-2):
                   kvy_r = kvy + ynum3
                   rows.append(kvy)
                   cols.append(kvy_r)
                   data.append(eta_s[i+1,j+1]/xstpc[j+1]/xstp[j])
               else:
                   rows.append(kvy)
                   cols.append(kvy)
                   data.append(B_right[i+1,3]*eta_s[i+1, j+1]/xstpc[j+1]/xstp[j])
                   R[kvy] -= B_right[i+1,2]*eta_s[i+1,j+1]/xstpc[j+1]/xstp[j]
                   
               # vx
               # top left vx node
               if (j > 0):
                   kvx_tl = kvy - 1 - ynum3
                   rows.append(kvy)
                   cols.append(kvx_tl)
                   data.append(eta_s[i+1,j]/ystpc[i+1]/xstp[j])
               else:
                   kvx_tr = kvy - 1
                   rows.append(kvy)
                   cols.append(kvx_tr)
                   data.append(B_left[i+1,1]*eta_s[i+1,j]/ystpc[i+1]/xstp[j])
                   R[kvy] -= B_left[i+1,0]*eta_s[i+1,j]/ystpc[i+1]/xstp[j]

               # bottom-left vx node
               if (j > 0):
                   kvx_bl = kvy - 1 + 3 - ynum3
                   rows.append(kvy)
                   cols.append(kvx_bl)
                   data.append(-eta_s[i+1,j]/ystpc[i+1]/xstp[j])
               else:
                   kvx_br = kvy - 1 + 3
                   rows.append(kvy)
                   cols.append(kvx_br)
                   data.append(-B_left[i+2,1]*eta_s[i+1,j]/ystpc[i+1]/xstp[j])
                   R[kvy] += B_left[i+2,0]*eta_s[i+1,j]/ystpc[i+1]/xstp[j]
               
               # top-right vx node
               if (j < xnum-2):
                   kvx_tr = kvy - 1
                   if (j > 0):
                       rows.append(kvy)
                       cols.append(kvx_tr)
                       data.append(-eta_s[i+1,j+1]/ystpc[i+1]/xstp[j])
                   else:
                       rows.append(kvy)
                       cols.append(kvx_tr)
                       data.append(-eta_s[i+1,j+1]/ystpc[i+1]/xstp[j])
               else:
                   kvx_tl = kvy - 1 - ynum3
                   rows.append(kvy)
                   cols.append(kvx_tl)
                   data.append(- B_right[i+1,1]*eta_s[i+1,j+1]/ystpc[i+1]/xstp[j])
                   R[kvy] += B_right[i+1,0]*eta_s[i+1,j+1]/ystpc[i+1]/xstp[j]

               # bottom right vx node
               if (j < xnum-2):
                   kvx_br = kvy - 1 + 3
                   if (j > 0):
                       rows.append(kvy)
                       cols.append(kvx_br)
                       data.append(eta_s[i+1,j+1]/ystpc[i+1]/xstp[j])
                   else:
                       rows.append(kvy)
                       cols.append(kvx_br)
                       data.append(eta_s[i+1,j+1]/ystpc[i+1]/xstp[j])
               else:
                   kvx_bl = kvy + 3 - 1 - ynum3
                   rows.append(kvy)
                   cols.append(kvx_bl)
                   data.append(B_right[i+2,1]*eta_s[i+1,j+1]/ystpc[i+1]/xstp[j])
                   R[kvy] += - B_right[i+2,0]*eta_s[i+1,j+1]/ystpc[i+1]/xstp[j]
               
               # Pressure
               # top P node
               kp_t = kvy + 1
               rows.append(kvy)
               cols.append(kp_t)
               data.append(Pscale/ystpc[i+1])
               # bottom P node
               kp_b = kvy + 1 + 3
               rows.append(kvy)
               cols.append(kp_b)
               data.append(-Pscale/ystpc[i+1])
               
           else:
               # at some kind of boundary
               rows.append(kvy)
               cols.append(kvy)
               data.append(2*Pscale/(xstp_av + ystp_av))
               if (j!=B_intern[4] or i<B_intern[5] or i>B_intern[6]):
                   # at the external boundary
                   R[kvy] = 0
               else:
                   # Internal prescribed horizontal velocity
                   R[kvy] = 2*Pscale/(xstp_av + ystp_av)*B_intern[7]
                   
                   
           ###################### continuity eqn ##############################
           if ( ((j>0 or i>0) and Bpres==0) or (i>0 and i<ynum-2 and Bpres==1) or (j>0 and j<xnum-2 and Bpres==2) ):
               
               # right part
               R[kP] = R_C[i,j]
               
               # left vx node
               if (j>0):
                   kvx_l = kP - 2 - ynum3
                   rows.append(kP)
                   cols.append(kvx_l)
                   data.append(-Pscale/xstp[j])
                   # Add boundary condition for the right Vx node
                   if (j==xnum-2):
                       rows.append(kP)
                       cols.append(kvx_l)
                       data.append(B_right[i+1,1]*Pscale/xstp[j])
                       R[kP] -= B_right[i+1,0]*Pscale/xstp[j]
               
               # right vx node
               if (j<xnum-2):
                   kvx_r = kP - 2
                   rows.append(kP)
                   cols.append(kvx_r)
                   data.append(Pscale/xstp[j])
                   # Add boundary condition for the right Vx node
                   if (j==0):
                       rows.append(kP)
                       cols.append(kvx_r)
                       data.append(-B_left[i+1,1]*Pscale/xstp[j])
                       R[kP] += B_left[i+1,0]*Pscale/xstp[j]
               
               # top vy node
               if (i>0):
                   kvy_t = kP - 1 - 3
                   rows.append(kP)
                   cols.append(kvy_t)
                   data.append(-Pscale/ystp[i])
                   # Add boundary condition for the bottom Vy node
                   if (i==ynum-2):
                       rows.append(kP)
                       cols.append(kvy_t)
                       data.append(B_bottom[j+1,3]*Pscale/ystp[i])
                       R[kP] -= B_bottom[j+1,2]*Pscale/ystp[i]
               
               # bottom vy node
               if (i<ynum-2):
                   kvy_b = kP - 1
                   rows.append(kP)
                   cols.append(kvy_b)
                   data.append(Pscale/ystp[i])
                   # Add boundary condition for the top Vy node
                   if (i==0):
                       rows.append(kP)
                       cols.append(kvy_b)
                       data.append(-B_top[j+1,3]*Pscale/ystp[i])
                       R[kP] += B_top[j+1,2]*Pscale/ystp[i]
           
           # pressure def for the boundary condition regions
           else:
               # pressure def in one cell
               if (Bpres==0):
                   rows.append(kP)
                   cols.append(kP)
                   data.append(2*Pscale/(xstp_av +ystp_av))
                   R[kP] = 2*Pnorm/(xstp_av +ystp_av)
               # pressure def at top and bottom
               if (Bpres==1):
                   rows.append(kP)
                   cols.append(kP)
                   data.append(2*Pscale/(xstp_av +ystp_av))
                   if (i==0):
                       R[kP] = 2*Pnorm/(xstp_av +ystp_av)
                   else:
                       R[kP] = 0
               # pressure def at left and right
               if (Bpres==2):
                   rows.append(kP)
                   cols.append(kP)
                   data.append(2*Pscale/(xstp_av +ystp_av))
                   if (j==0):
                       R[kP] = 2*Pnorm/(xstp_av +ystp_av)
                   else:
                       R[kP] = 0
    
    return rows, cols, data, R



def StokesContinuitySolver(P_first, eta_s, eta_n, xnum, ynum, gridx, gridy, R_x, R_y, R_C, B_top, B_bottom, B_left, B_right, B_intern):
    '''
    This function formulates and solves  
    Stokes and Continuity equations defined on 2D staggered irregularly spaced grid
    with specified resolution (xnum, ynum) and grid lines positions (gridx, gridy)
    given distribution of right parts for all equations (RX,RY,RC) on the grid 
    and given variable shear (etas) and normal (etan) viscosity distributions 
    pressure is normalized relative to given value (prnorm) in the first cell

    Parameters
    ----------
    P_first : ARRAY
        2 element array that defines pressure BCs. 
        P_first[0] sets the type of pressure BC (0=defined by one cell, 1=top and bottom).
        P_first[1] is the boundary value
    eta_s : ARRAY
        The shear stress viscosity values.
    eta_n : ARRAY
        The normal stress viscosity values.
    xnum : INT
        Number of normal nodes in x-direction.
    ynum : INT
        Number of normal nodes in y-direction.
    gridx : ARRAY
        x-coordinates of normal grid nodes.
    gridy : ARRAY
        y-coordinates of normal grid nodes.
    R_x : ARRAY
        RHS values of the x-component momentum equation.
    R_y : ARRAY
        RHS values of the y-component momentum equation
    R_C : ARRAY
        RHS values of the continuity equation
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
        B_intern[3] = prescribed x-velocity value.
        B_intern[4] = x-index of vy nodes with prescribed velocity (-1 is not in use)
        B_intern[5-6] = min/max y-index of the wall
        B_intern[7] = prescribed y-velocity value.

    Returns
    -------
    vx : ARRAY
        Calculated vx values.
    vy : ARRAY
        Calculated vy values.
    P : ARRAY
        Calculated pressure values.
    resx : ARRAY
        residuals for the vx component.
    resy : ARRAY
        residuals for the vy component.
    resc : ARRAY
        residuals for the continuity eqn.

    '''
    
    # pressure bcs?
    Bpres = 0
    Pnorm = P_first[1]
    # Channel flow top->bottom
    if (P_first[0]==1):
        Bpres = 1
        Pnorm=P_first[1]
    
    # grid steps for the basic nodes
    xstp = gridx[1:] - gridx[:-1]
    ystp = gridy[1:] - gridy[:-1]
    
    
    # grid steps for the vx and vy nodes
    xstpc = np.zeros((xnum))
    ystpc = np.zeros((ynum))
    
    xstpc[0] = xstp[0]
    ystpc[0] = ystp[0]
    
    xstpc[xnum-1] = xstp[xnum-2]
    ystpc[ynum-1] = ystp[ynum-2]
    
    xstpc[1:xnum-1] = (gridx[2:] - gridx[:xnum-2])/2
    ystpc[1:ynum-1] = (gridy[2:] - gridy[:ynum-2])/2
    
    # average x and y steps
    xstp_av = (gridx[xnum-1] - gridx[0])/(xnum-1)
    ystp_av = (gridy[ynum-1] - gridy[0])/(ynum-1)
    
    # coefficient of pressure scaling
    Pscale = 2*eta_n[0,0]/(xstp_av+ystp_av)
    
    # construct the matrix system
    rows, cols, dL, R = StokesConstructMatrix(xnum, ynum, xstp, xstpc, ystp, ystpc, xstp_av, ystp_av, Pscale, Pnorm, Bpres,\
                                 eta_s, eta_n, R_x, R_y, R_C, B_top, B_bottom, B_left, B_right, B_intern)
    
    ###########################################################################
    # solve the matrix system
    
    # first construct the sparse matrix
    N = (xnum-1)*(ynum-1)*3
    L = coo_array((dL, (rows,cols)), shape=(N,N)).tocsr()
    
    # using scipy's spsolve for sparse matricies
    S = spsolve(L,R)
    
    # transform result back to spatial grid
    vx, vy, P = S_to_grid_Stokes(S, xnum, ynum, Pscale, B_top, B_bottom, B_left, B_right)
    
    ###########################################################################
    # residuals
    resx, resy, resc = calculateResiduals(xnum, ynum, xstp, xstpc, ystp, ystpc, eta_s, eta_n, vx, vy, P, R_x, R_y, R_C, B_intern)
    

    return vx, vy, P, resx, resy, resc

@jit
def calculateResiduals(xnum, ynum, xstp, xstpc, ystp, ystpc, eta_s, eta_n, vx, vy, P, R_x, R_y, R_C, B_intern):
    '''
    Calculates the residuals of the Stoke's solver.

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
    eta_s : ARRAY
        The shear stress viscosity values.
    eta_n : ARRAY
        The normal stress viscosity values.
    vx : ARRAY
        x velocities.
    vy : ARRAY
        y velocities
    P : ARRAY
        Calculated pressure values.
    R_x : ARRAY
        RHS values of the x-component momentum equation.
    R_y : ARRAY
        RHS values of the y-component momentum equation
    R_C : ARRAY
        RHS values of the continuity equation
    B_intern : ARRAY
        Array defining optional internal boundary eg. moving wall.

    Returns
    -------
    resx : ARRAY
        x-momentum eqn residuals.
    resy : ARRAY
        y-momentum eqn residuals.
    resc : ARRAY
        Continuity eqn residuals.

    '''
    
    # initialize arrays
    resx = np.zeros((ynum+1, xnum))
    resy = np.zeros((ynum, xnum+1))
    resc = np.zeros((ynum-1, xnum-1))
    
    # calculate residuals
    for i in range(0, ynum+1):
        for j in range(0, xnum+1):
            ####################################################################
            # x Stokes eqn
            if (j<xnum and (j!=B_intern[0] or i<B_intern[1] or i>B_intern[2])):
                if (i==0 or i==ynum or j==0 or j==xnum-1):
                    resx[i,j] = 0
                else:
                    resx[i,j] = R_x[i,j] - (2*(eta_n[i-1,j]*(vx[i,j+1] - vx[i,j])/xstp[j]\
                                               - eta_n[i-1,j-1]*(vx[i,j] - vx[i,j-1])/xstp[j-1])\
                                            - (P[i-1,j] - P[i-1,j-1]))/xstpc[j]
    
                    resx[i,j] += -(eta_s[i,j]*((vx[i+1,j] - vx[i,j])/ystpc[i] + (vy[i,j+1] - vy[i,j])/xstpc[j])\
                                   -eta_s[i-1,j]*((vx[i,j] - vx[i-1,j])/ystpc[i-1] + (vy[i-1,j+1] - vy[i-1,j])/xstpc[j]))/ystp[i-1]

        
            ####################################################################
            # y Stokes eqn
            if (i<ynum and (j!=B_intern[4] or i<B_intern[5] or i>B_intern[6])):
                if (i==0 or i==ynum-1 or j==0 or j==xnum):
                    resy[i,j] = 0
                else:
                    resy[i,j] = R_y[i,j] - (2*(eta_n[i,j-1]*(vy[i+1,j] - vy[i,j])/ystp[i]\
                                               - eta_n[i-1,j-1]*(vy[i,j] - vy[i-1,j])/ystp[i-1])\
                                            - (P[i,j-1] - P[i-1,j-1]))/ystpc[i]
    
                    resy[i,j] += -(eta_s[i,j]*((vy[i,j+1] - vy[i,j])/xstpc[j] + (vx[i+1,j] - vx[i,j])/ystpc[i])\
                                   -eta_s[i,j-1]*((vy[i,j] - vy[i,j-1])/xstpc[j-1] + (vx[i+1,j-1] - vx[i,j-1])/ystpc[i]))/xstp[j-1]

    
            ###################################################################
            # continuity eqn
            if (i<ynum-1 and j<xnum-1):
                resc[i,j] = R_C[i,j] - ((vx[i+1,j+1] - vx[i+1,j])/xstp[j] + (vy[i+1,j+1] - vy[i,j+1])/ystp[i])
                
    return resx, resy, resc
                
                
