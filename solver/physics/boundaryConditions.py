#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boundary Conditions

"""

import numpy as np
from numba import float64, int64
from numba.experimental import jitclass


spec_BC = [
    ('xnum', int64),
    ('ynum', int64),
    ('P_first', float64[:]),
    ('B_top', float64[:,:]),
    ('B_bottom', float64[:,:]),
    ('B_left', float64[:,:]),
    ('B_right', float64[:,:]),
    ('B_intern', float64[:]),
    ('BT_top', float64[:,:]),
    ('BT_bottom', float64[:,:]),
    ('BT_left', float64[:,:]),
    ('BT_right', float64[:,:])
    ]

@jitclass(spec_BC)
class BCs():
    """
    Class which stores the arrays defining the boundary conditions for velocity, temperature, pressure
    and the internal wall velocity boundary.  
    
    These values can be set manually, but this class also provides helper functions 
    to set common conditions.  See the docstrings for these for details on the 
    available conditions.
    
    Attributes
    ----------
    P_first : ARRAY
        Array with 2 entries, specifying pressure BC.
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
    BT_top : ARRAY
        Top temperature BCs.  Array has 2 columns, values in each are defined as:
        T[i,j] = BT_top[0] + BT_top[1]*T[i+1,j]
    BT_bottom : ARRAY
        Bottom temperature BCs.  Array has 2 columns, values in each are defined as:
        T[i,j] = BT_bottom[0] + BT_bottom[1]*T[i-1,j]
    BT_left : ARRAY
        Left temperature BCs.  Array has 2 columns, values in each are defined as:
        T[i,j] = BT_left[0] + BT_left[1]*T[i,j+1]
    BT_right : ARRAY
        Right temperature BCs.  Array has 2 columns, values in each are defined as:
        T[i,j] = BT_right[0] + BT_right[1]*T[i,j-1]
    """
    
    
    def __init__(self, xnum, ynum):
        """
        Constructs the BCs object, initializes the arrays to be zeroed, user must 
        then populate with correct values using either this class' helper functions
        or by manually setting the array values.

        Parameters
        ----------
        xnum : INT
            Number of grid points in x-direction.
        ynum : INT
            Number of grid points in y-direction.

        Returns
        -------
        None.

        """
        self.xnum = xnum
        self.ynum = ynum
        
        # pressure BC
        self.P_first = np.array([0.0,0.0])
        
        # velocity BCs
        self.B_top = np.zeros((xnum+1, 4))
        self.B_bottom = np.zeros((xnum+1, 4))
        self.B_left = np.zeros((ynum+1, 4))
        self.B_right = np.zeros((ynum+1, 4))

        # optional internal boundary, initialised to be switched off
        self.B_intern = np.zeros(8)
        self.B_intern[0] = -1.0
        self.B_intern[4] = -1.0
        
        
        # Temperature BCs
        self.BT_top = np.zeros((xnum,2))
        self.BT_bottom = np.zeros((xnum, 2))
        self.BT_left = np.zeros((ynum, 2))
        self.BT_right = np.zeros((ynum, 2))

    
    def set_top_BC(self, condition, v=None):
        """
        sets the top velocity boundary condition to one of: no slip, free slip or prescribed parallel velocity.
        

        Parameters
        ----------
        condition : STR
            Name of the condition to be set, options are "no slip", "free slip" or "prescribed parallel velocity".
        v : FLOAT, optional
            If the "presribed parallel velocity" option is used, this is the required velocity. The default is None.

        Returns
        -------
        None.

        """
        
        
        if condition=="no slip":
            self.B_top = np.zeros(np.shape(self.B_top))
            
        elif condition=="free slip":
            self.B_top[:,1] = 1.0
        
        elif condition=="prescribed parallel velocity":
            if v==None:
                raise ValueError("You must provide a velocity value if using prescribed velocity conditions")
            else:
                self.B_top[:,0] = v
            
        else:
            raise ValueError("Boundary condition option not recognised, run show_BC_options to see available BC types")
    
        
    def set_bottom_BC(self, condition, v=None):
        """
        sets the bottom velocity boundary condition to one of: no slip, free slip or prescribed parallel velocity.
        

        Parameters
        ----------
        condition : STR
            Name of the condition to be set, options are "no slip", "free slip" or "prescribed parallel velocity".
        v : FLOAT, optional
            If the "presribed parallel velocity" option is used, this is the required velocity. The default is None.

        Returns
        -------
        None.

        """
        
        
        if condition=="no slip":
            self.B_bottom = np.zeros(np.shape(self.B_bottom))
            
        elif condition=="free slip":
            self.B_bottom[:,1] = 1.0
        
        elif condition=="prescribed parallel velocity":
            if v==None:
                raise ValueError("You must provide a velocity value if using prescribed velocity conditions")
            else:
                self.B_bottom[:,0] = v
            
        else:
            raise ValueError("Boundary condition option not recognised, run show_BC_options to see available BC types")
        
    
    def set_left_BC(self, condition, v=None):
        """
          sets the left velocity boundary condition to one of: no slip, free slip or prescribed parallel velocity.
          

          Parameters
          ----------
          condition : STR
              Name of the condition to be set, options are "no slip", "free slip" or "prescribed parallel velocity".
          v : FLOAT, optional
              If the "presribed parallel velocity" option is used, this is the required velocity. The default is None.

          Returns
          -------
          None.

          """
          
          
        if condition=="no slip":
            self.B_left = np.zeros(np.shape(self.B_left))
              
        elif condition=="free slip":
            self.B_left[:,3] = 1.0
          
        elif condition=="prescribed parallel velocity":
            if v==None:
                raise ValueError("You must provide a velocity value if using prescribed velocity conditions")
            else:
                self.B_left[:,2] = v
              
        else:
            raise ValueError("Boundary condition option not recognised, run show_BC_options to see available BC types")
            

    def set_right_BC(self, condition, v=None):
        """
        sets the right velocity boundary condition to one of: no slip, free slip or prescribed parallel velocity.
        

        Parameters
        ----------
        condition : STR
            Name of the condition to be set, options are "no slip", "free slip" or "prescribed parallel velocity".
        v : FLOAT, optional
            If the "presribed parallel velocity" option is used, this is the required velocity. The default is None.

        Returns
        -------
        None.

        """
        
        
        if condition=="no slip":
            self.B_right = np.zeros(np.shape(self.B_right))
            
        elif condition=="free slip":
            self.B_right[:,3] = 1.0
        
        elif condition=="prescribed parallel velocity":
            if v==None:
                raise ValueError("You must provide a velocity value if using prescribed velocity conditions")
            else:
                self.B_right[:,2] = v
            
        else:
            raise ValueError("Boundary condition option not recognised, run show_BC_options to see available BC types")
        
    
    ###########################################################################
    # temperature BC setters
    
    def set_T_BC(self, BC, condition, T=None):
        """
        sets a temperature boundary condition.  Options are "insulating" or "fixed T"

        For consistency with the velocity conditions, this function is used by wrappers for the specific top, bot, etc.
        directions, but since the implementation is the same we can use this one function for all.

        Parameters
        ----------
        BC : ARRAY
            The chosen temperature boundary array, should be a member of the BC object.
        condition : STR
            Boundary condition choice, options are "insulating" or "fixed T".
        T : FLOAT, optional
            If fixed temperature condition is chosen, this variable should specify the temperature. The default is None.

        Returns
        -------
        None.

        """
        
        if condition=="insulating":
            BC[:,1] = 1.0
            
        elif condition=="fixed T":
            if T==None:
                raise ValueError("You must provide a temperature if using fixed T boundary conditions")
            else:
                BC[:,0] = T
        
    
    def set_top_T_BC(self, condition, T=None):
        """
        sets the top (y=0) temperature boundary.  Options are "insulating" or "fixed T"

        Parameters
        ----------
        condition : STR
            Boundary condition choice, options are "insulating" or "fixed T".
        T : FLOAT, optional
            If fixed temperature condition is chosen, this variable should specify the temperature. The default is None.

        Returns
        -------
        None.

        """
        
        self.set_T_BC(self.BT_top, condition, T=T)
        
    
    def set_bottom_T_BC(self, condition, T=None):
        """
        sets the bottom (y=ysize) temperature boundary.  Options are "insulating" or "fixed T"

        Parameters
        ----------
        condition : STR
            Boundary condition choice, options are "insulating" or "fixed T".
        T : FLOAT, optional
            If fixed temperature condition is chosen, this variable should specify the temperature. The default is None.

        Returns
        -------
        None.

        """
        
        self.set_T_BC(self.BT_bottom, condition, T=T)
    
        
    def set_left_T_BC(self, condition, T=None):
        """
        sets the left (x=0) temperature boundary.  Options are "insulating" or "fixed T"

        Parameters
        ----------
        condition : STR
            Boundary condition choice, options are "insulating" or "fixed T".
        T : FLOAT, optional
            If fixed temperature condition is chosen, this variable should specify the temperature. The default is None.

        Returns
        -------
        None.

        """
        
        self.set_T_BC(self.BT_left, condition, T=T)
        
    def set_right_T_BC(self, condition, T=None):
        """
        sets the right (x=xsize) temperature boundary.  Options are "insulating" or "fixed T"

        Parameters
        ----------
        condition : STR
            Boundary condition choice, options are "insulating" or "fixed T".
        T : FLOAT, optional
            If fixed temperature condition is chosen, this variable should specify the temperature. The default is None.

        Returns
        -------
        None.

        """
        
        self.set_T_BC(self.BT_right, condition, T=T)
        