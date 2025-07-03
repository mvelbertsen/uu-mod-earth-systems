#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions that relate to marker physics,
including interpolation of quantities to/from the grid,
marker viscosity calculation, subgrid stress/diffusion

Lower level functions for marker operations, including getting nearest node 
distances, interpolating a value to/from markers, etc. are in markerUtils.

"""
import numpy as np
from numba import jit

from dataStructures import Markers, Materials, Grid
from physics.markerUtils import applyGridContrib, applyMarkerContrib,\
    getMarkerNodeDistances, findNearestNode, applyGridWeights


@jit(nopython=True)
def markersToGrid(markers, materials, grid, grid0, xnum, ynum, params, tstep, ntstp, plast_y):
    '''
    Reads or calculates the marker values for density, thermal quantities, viscosity, stresses, mu
    and interpolates them to the grid.  This is called at the beginning of each timestep to update the
    grid with the advected marker values from the end of the previous step. 

    Parameters
    ----------
    markers : Markers
        Markers object containing all the marker variables.
    materials : Materials
        Materials object containing values of material properties for each material in use.
    grid : Grid
        Grid object containing all the grid variables, will be updated by this function.
    grid0 : Grid
        Grid object containing all the previous step's grid variables.
    xnum : INT
        x-resolution of the simulation domain.
    ynum : INT
        y-resolution of the simulation domain.
    params : Parameters
        Parameters object containing all the simulation parameters.
    tstep : FLOAT
        Current timestep value.
    ntstp : INT
        Current timestep number.
    plast_y : INT
        Flag to indicate whether plastic yielding has occured, will be updated by this function.

    Returns
    -------
    None.

    '''
    
    # set local variable for grid items
    gridx = grid.x
    gridy = grid.y
    xstp = grid.xstp
    ystp = grid.ystp
    
    
    for m in range(0,markers.num):
        # check that the marker is inside the grid
        if (markers.x[m]>= gridx[0] and markers.x[m]<=gridx[xnum-1] and markers.y[m]>=gridy[0] and markers.y[m]<=gridy[ynum-1]):
            
            # find the indicies of the top-left node of the cell the current marker is in
            xn, yn = findNearestNode(gridx, gridy, xnum, ynum, markers.x[m], markers.y[m])
            # save indicies
            markers.nx[m] = xn
            markers.ny[m] = yn
            
            # compute distance to i,j node
            dxm = (markers.x[m] - gridx[xn])/xstp[xn]
            dym = (markers.y[m] - gridy[yn])/ystp[yn]
        
            # set the marker weight factor, not in use
            mwt = 1
            
            # compute the density from the marker temperature
            m_rho = materials.rho[markers.id[m],0]*(1 - materials.rho[markers.id[m],1]*(markers.T[m]-273))\
                         *(1 + materials.rho[markers.id[m],2]*(markers.P[m] - 1e+5)) #TODO: remove this hard-coded value!
            
            # compute rhoCP for the marker
            m_rhoCP = m_rho*materials.Cp[markers.id[m]]
            
            # compute thermal conductivity
            m_KT = materials.kT[markers.id[m],0] + materials.kT[markers.id[m],1]/(markers.T[m] + 77) #TODO: hard-coded value
            
            # compute adiabatic heating 
            m_HA = materials.rho[markers.id[m],1]*markers.T[m]
            
            # add properties to the surrounding nodes
            applyMarkerContrib(grid.rho, m_rho, dxm, dym, xn, yn, mwt)
            applyMarkerContrib(grid.rhoCP, m_rhoCP, dxm, dym, xn, yn, mwt)
            applyMarkerContrib(grid.T, markers.T[m], dxm, dym, xn, yn, mwt)
            applyMarkerContrib(grid.kT, m_KT, dxm, dym, xn, yn, mwt)
            applyMarkerContrib(grid.H_a, m_HA, dxm, dym, xn, yn, mwt)
            applyMarkerContrib(grid.H_r, materials.radH[markers.id[m]], dxm, dym, xn, yn, mwt)
            applyMarkerContrib(grid.wt, 1.0, dxm, dym, xn, yn, mwt)
            
            
            ###################################################################
            # compute marker viscosity
            m_eta = markerViscosity(markers, materials, m, grid, params, ntstp, tstep, plast_y)
            
            # compute 1/mu
            m_mu = 1/materials.mu[markers.id[m]]
            
            # apply marker viscosity to nodes, only in surrounding halfcell
            applyMarkerContrib(grid.eta_s, m_eta, dxm, dym, xn, yn, mwt, width=0.5)
            applyMarkerContrib(grid.mu_s, m_mu, dxm, dym, xn, yn, mwt, width=0.5)
            applyMarkerContrib(grid.sigxy, markers.sigmaxy[m], dxm, dym, xn, yn, mwt, width=0.5)
            applyMarkerContrib(grid.wt_eta_s, 1.0, dxm, dym, xn, yn, mwt, width=0.5) # need bc of discrepancy in no of markers
            
            # add normal visc, sxx, shear modulus to center of current cell            
            wij = (1-abs(0.5-dxm))*(1-abs(0.5-dym)) # i,j node
            
            grid.eta_n[yn,xn] += wij*m_eta*mwt
            grid.mu_n[yn,xn] += wij*m_mu*mwt
            grid.sigxx[yn,xn] += wij*markers.sigmaxx[m]*mwt
            grid.wt_eta_n[yn, xn] += wij*mwt
                
    # finally, use weights to calculate the new grid values
    applyGridWeights(xnum, ynum, grid, grid0)

    
@jit(nopython=True)
def markerViscosity(markers, materials, m, grid, params, ntstp, tstep, plast_y):
    '''
    Calculates the viscosity of a marker based on its material type.

    Parameters
    ----------
    markers : Markers
        Markers object containing all the marker variables.
    materials : Materials
        Materials object containing values of material properties for each material in use.
    m : INT
        Index of the marker to be calculated for.
    grid : Grid
        Grid object containing all the grid variables, will be updated by this function.
    params : Parameters
        Parameters object containing all the simulation parameters.
    tstep : FLOAT
        Current timestep value.
    ntstp : INT
        Current timestep number.
    plast_y : INT
        Flag to indicate whether plastic yielding has occured, will be updated by this function.

    Yields
    ------
    m_eta : FLOAT
        Calculated marker viscosity value.

    '''
    mID = markers.id[m]
    
    if (materials.visc[mID, 0] < 1e-11):
        # constant viscosity
        m_eta = materials.visc[mID, 1]
        # High constant viscosity at internal boundary 
    elif (markers.x[m] > 430000 and markers.x[m] < 550000 and markers.y[m] > 30000 and markers.y[m] < 60000):
        m_eta = 1e28
    else:
        # power law viscosity
        # eps_ii = Ad * sigma_ii^n * exp(-(Ea+Va*P)/RT)
        
        # get old deviatoric stress
        sig_ii0 = np.sqrt(markers.sigmaxx[m]**2 + markers.sigmaxy[m]**2)
        
        # check against limit
        if (sig_ii0 < params.stress_min):
            sig_ii0 = params.stress_min
        
        # check marker temp
        T_pow_law = markers.T[m]
        if (T_pow_law < params.T_min):
            T_pow_law = params.T_min
        
        # compute exponential term
        pow_law_exp = (materials.visc[mID,4] + materials.visc[mID,5]*markers.P[m])/(params.Rgas*T_pow_law)

        # cut off if too big (at cold T)
        if (pow_law_exp > params.max_pow_law):
            pow_law_exp = params.max_pow_law
        
        # compute Ad*exp(-Ea + Va*P/RT)
        ad_pow_law_exp = materials.visc[mID,2] * np.exp(-pow_law_exp)
        
        # compute strain rate from power law
        eps_ii0 = ad_pow_law_exp*(1e-6*sig_ii0)**materials.visc[mID,3]
        
        # compute effective visc
        eta0 = sig_ii0/(2*eps_ii0)
        
        # compute second viscosity value or marker.eta 
        x_elvis = eta0/(materials.mu[mID]*tstep + eta0)
        
        sig_xx_new = markers.sigmaxx[m]*x_elvis + 2*eta0*markers.epsxx[m]*markers.E_rat[m]*(1 - x_elvis)
        sig_xy_new = markers.sigmaxy[m]*x_elvis + 2*eta0*markers.epsxy[m]*markers.E_rat[m]*(1 - x_elvis)
        
        sig_ii1 = np.sqrt(sig_xx_new**2 + sig_xy_new**2)

        # check against limit
        if (sig_ii1 < params.stress_min):
            sig_ii1 = params.stress_min
        
        # compute strain rate from power late
        eps_ii1 = ad_pow_law_exp*(1e-6*sig_ii1)**materials.visc[mID,3]
        
        # compute effective visco
        m_eta = sig_ii1/(2*eps_ii1)
        
        # iterate for viscosity which corresponds to future stress invariant
        pow_law_it = 0
        while (pow_law_it < 20 and abs(sig_ii1 - sig_ii0) > 1):
            # update counter
            pow_law_it += 1
            
            # compute middle stress
            sig_ii_mid = (sig_ii0 + sig_ii1)/2
            
            eps_ii_mid = ad_pow_law_exp*(1e-6*sig_ii_mid)**materials.visc[mID,3]
            
            m_eta = sig_ii_mid/(2*eps_ii_mid)
            
            x_elvis = m_eta/(materials.mu[mID]*tstep + m_eta)
            
            sigxx_new = markers.sigmaxx[m]*x_elvis + 2*m_eta*markers.epsxx[m]*markers.E_rat[m]*(1 - x_elvis)
            sigxy_new = markers.sigmaxy[m]*x_elvis + 2*m_eta*markers.epsxy[m]*markers.E_rat[m]*(1 - x_elvis)
            
            sig_ii_new = np.sqrt(sigxx_new**2 + sigxy_new**2)
            
            # changing bisection limits
            if ( (sig_ii0 < sig_ii1 and sig_ii_mid<sig_ii_new) or (sig_ii0 > sig_ii1 and sig_ii_mid>sig_ii_new) ):
                sig_ii0 = sig_ii_mid
            else:
                sig_ii1 = sig_ii_mid
        
        # apply limiting
        if (m_eta < params.eta_min):
            m_eta = params.eta_min
        if (m_eta > params.eta_max):
            m_eta = params.eta_max
        
    # check if any plastic yielding condition is present
    if (ntstp > 0 and (materials.plast[mID,0] > 0 or materials.plast[mID,2]>0)):
        # check if there is plastic yielding
        sigxx_new_e = markers.sigmaxx[m] + 2*materials.mu[mID]*tstep*markers.epsxx[m]*markers.E_rat[m]
        sigxy_new_e = markers.sigmaxy[m] + 2*materials.mu[mID]*tstep*markers.epsxy[m]*markers.E_rat[m]
        sigii_new_e = np.sqrt(sigxx_new_e**2 + sigxy_new_e**2)
        
        # checking yielding criterion for strain weakening/hardening
        m_cohes = materials.plast[mID,0]
        m_frict = materials.plast[mID,2]
        if (markers.gII[m] >= materials.plast[mID,5]):
            m_cohes = materials.plast[mID,1]
            m_frict = materials.plast[mID,3]
        
        if (markers.gII[m] > materials.plast[mID,4] and markers.gII[m] < materials.plast[mID,5]):
            m_cohes = materials.plast[mID,0] + (materials.plast[mID,1]- materials.plast[mID,0])\
                        /(materials.plast[mID,5] - materials.plast[mID,4])*(markers.gII[m] - materials.plast[mID,4])
            m_frict = materials.plast[mID,2] + (materials.plast[mID,3]- materials.plast[mID,2])\
                        /(materials.plast[mID,5] - materials.plast[mID,4])*(markers.gII[m] - materials.plast[mID,4])
            
        # computing yield stress of marker
        sii_yield = m_cohes + m_frict*markers.P[m]
        if (sii_yield < 0):
            sii_yield = 0
        
        # correcting rock props for yielding
        if (sii_yield < sigii_new_e):
            # bring marker visc to yield stress
            m_etap = materials.mu[mID]*tstep*sii_yield/(sigii_new_e - sii_yield)
            m_etap = m_etap**(1-params.eta_wt)*markers.eta[m]**params.eta_wt
            if (m_etap < m_eta):
                m_eta = m_etap
                # limiting visc for yielding
                if (m_eta < params.eta_min):
                    m_eta = params.eta_min
                elif (m_eta > params.eta_max):
                    m_eta = params.eta_max
                # mark that yielding occurs
                plast_y = 1
                
    # save marker viscosity
    markers.eta[m] = m_eta
    
    return m_eta
    



@jit(nopython=True)
def gridToMarker(fields, markers_fs, markersx, markersy, markersnx, markersny, grid, node_type=0):
    '''
    Interpolates grid values to markers for a specified list of variables

    Parameters
    ----------
    fields : LIST
        List containing the grid variable arrays to be interpolated.  If only a single variable,
        it must still be in a list, e.g. [grid.rho].
    markers_fs : LIST
        List containing the marker variable arrays to be interpolated to.  If only a single variable,
        it must still be in a list, e.g. [markers.T].
    markersx : ARRAY
        Marker x-coordinates.
    markersy : ARRAY
        Marker y-coordinates.
    markersnx : ARRAY
        x-index of nearest top-left basic node for markers.
    markersny : ARRAY
        y-index of nearest top-left basic node for markers.
    grid : Grid
        Grid object containing all the grid variables
    node_type : INT, optional
        Specifies which type of nodes the variables interpolated come from. The default is 0 = basic nodes.
        1 = pressure nodes.

    Returns
    -------
    None.

    '''

    for m in range(0,len(markers_fs[0])):
        # check that the marker is inside the grid
        if (markersx[m]>= grid.x[0] and markersx[m]<=grid.x[len(grid.x)-1] and markersy[m]>=grid.y[0] and markersy[m]<=grid.y[len(grid.y)-1]):
            
            dxm, dym, xn, yn = getMarkerNodeDistances(markersx[m], markersy[m], markersnx[m], markersny[m], len(grid.x), len(grid.y), grid, node_type)
            
            # apply contributions from surrounding four nodes for each field
            for f in range(0,len(fields)):
                field = fields[f]
                markers_fs[f][m] = applyGridContrib(field, xn, yn, dxm, dym)
                
            

@jit(nopython=True)
def updateMarkerErat(markers, materials, grid, timestep):
    '''
    Updates the marker eii/eii_grid ratio values

    Parameters
    ----------
    markers : Markers
        Markers object containing all the marker variables.
    materials : Materials
        Materials object containing values of material properties for each material in use.
    grid : Grid
        Grid object containing all the grid variables, will be updated by this function.
    timestep : FLOAT
        Current timestep value.

    Returns
    -------
    None.

    '''
    
    for m in range(0, markers.num):
        # check that the marker is inside the grid
        if (markers.x[m]>= grid.x[0] and markers.x[m]<=grid.x[len(grid.x)-1] and markers.y[m]>=grid.y[0] and markers.y[m]<=grid.y[len(grid.y)-1]):
         
            # calcuate marker stresses
            dxm, dym, xn, yn = getMarkerNodeDistances(markers.x[m], markers.y[m], markers.nx[m], markers.ny[m], len(grid.x), len(grid.y), grid, 0)
            sxym = applyGridContrib(grid.sigxy2, xn, yn, dxm, dym)
            dsxym = applyGridContrib(grid.dsigxy, xn, yn, dxm, dym)
            
            # and at pressure nodes
            dxm, dym, xn, yn = getMarkerNodeDistances(markers.x[m], markers.y[m], markers.nx[m], markers.ny[m], len(grid.x), len(grid.y), grid, 1)
            sxxm = applyGridContrib(grid.sigxx2, xn, yn, dxm, dym)
            dsxxm = applyGridContrib(grid.dsigxx, xn, yn, dxm, dym)
            
            
            # calculate the marker strain rate  

            eii_mg = np.sqrt(markers.epsxx[m]**2 + markers.epsxy[m]**2)
            if (eii_mg > 0):
                # correct second strain rate using Maxwell model
                eii_m = np.sqrt( (sxxm/(2*markers.eta[m]) + dsxxm/(2*timestep*materials.mu[markers.id[m]]))**2\
                                + (sxym/(2*markers.eta[m]) + dsxym/(2*timestep*materials.mu[markers.id[m]]))**2)
                markers.E_rat[m] = eii_m/eii_mg
            else:
                markers.E_rat[m] = 1

    
@jit(nopython=True)
def subgridStressChanges(markers, grid, xnum, ynum, materials, params, timestep):
    '''
    Calculates the subgrid stress changes, stored in grid.dsigxx, grid.dsigxy

    Parameters
    ----------
    markers : Markers
        Markers object containing all the marker variables.
    grid : Grid
        Grid object containing all the grid variables, will be updated by this function.
    xnum : INT
        x-resolution of the simulation domain.
    ynum : INT
        y-resolution of the simulation domain.
    materials : Materials
        Materials object containing values of material properties for each material in us
    params : Parameters
        Parameters object containing all the simulation parameters.
    timestep : FLOAT
        Current timestep value.

    Returns
    -------
    None.

    '''
    
    dsxyn = np.zeros((ynum, xnum))
    dsxxn = np.zeros((ynum-1, xnum-1))
    
    grid.wt_eta_s = np.zeros((ynum, xnum))
    grid.wt_eta_n = np.zeros((ynum-1, xnum-1))
    
    for m in range(0,markers.num):
        # check that the marker is inside the grid
        if (markers.x[m]>= grid.x[0] and markers.x[m]<=grid.x[xnum-1] and markers.y[m]>=grid.y[0] and markers.y[m]<=grid.y[ynum-1]):
            
            # compute local stress relaxation timescale
            sdm = markers.eta[m]/materials.mu[markers.id[m]]
            
            # computing degree of subgrid stress relaxation
            sdif = -params.dsubgrid*timestep/sdm
            if (sdif < -30):
                sdif = -30
            
            sdif = 1 - np.exp(sdif)
            
            # get the node-grid separations
            dxm, dym, xn, yn = getMarkerNodeDistances(markers.x[m], markers.y[m], markers.nx[m], markers.ny[m], xnum, ynum, grid, 0)
            
            # interpolate old stress for the marker
            sxym = applyGridContrib(grid.sigxy, xn, yn, dxm, dym)
            
            # calculate nodal-marker subgrid sxy
            dsxym = sxym - markers.sigmaxy[m]
            # relax nodal-marker subgrid stress
            dsxym = dsxym*sdif
            
            # correcting old stress for the marker
            markers.sigmaxy[m] += dsxym
            
            mwt = 1.0
            
            # interpolate changes back to grid
            applyMarkerContrib(dsxyn, dsxym, dxm, dym, xn, yn, mwt, width=0.5)
            applyMarkerContrib(grid.wt_eta_s, 1, dxm, dym, xn, yn, mwt, width=0.5)
            
            # computing marker wieght for centre of current cell
            mwt = mwt*(1 - abs(0.5-dxm))*(1-abs(0.5-dym))
            
            # get the center node grid sep
            dxm, dym, xn, yn = getMarkerNodeDistances(markers.x[m], markers.y[m], markers.nx[m], markers.ny[m], xnum, ynum, grid, 1)
            
            # interpolate old stress for the marker
            sxxm = applyGridContrib(grid.sigxx, xn, yn, dxm, dym)
            
            # calculate nodal-marker subgrid sxy
            dsxxm = sxxm - markers.sigmaxx[m]
            # relax nodal-marker subgrid stress
            dsxxm = dsxxm*sdif
            
            # correcting old stress for the marker
            markers.sigmaxx[m] += dsxxm
            
            # interpolate changes back to center node of current cell
            dsxxn[markers.ny[m], markers.nx[m]] += dsxxm*mwt
            grid.wt_eta_n[markers.ny[m], markers.nx[m]] += mwt
            
            
    # compute subgrid stress changes for nodes
    for i in range(0,ynum):
        for j in range(0,xnum):
            if (grid.wt_eta_s[i,j] >= 1e-7):
                dsxyn[i,j] = dsxyn[i,j]/grid.wt_eta_s[i,j]
            
            if (i<ynum-1 and j<xnum-1 and grid.wt_eta_n[i,j] >= 1e-7):
                dsxxn[i,j] = dsxxn[i,j]/grid.wt_eta_n[i,j]
    
    # subtracting subgrid stress change from nodal stress changes
    grid.dsigxy = grid.dsigxy - dsxyn
    grid.dsigxx = grid.dsigxx - dsxxn
    
    
@jit(nopython=True)
def subgridDiffusion(grid, markers, params, xnum, ynum, timestep, xstp_av, ystp_av):
    '''
    Calculates the subgrid diffusion temperature changes for all grid nodes.

    Parameters
    ----------
    grid : Grid
        Grid object containing all the grid variables.
    markers : Markers
        Markers object containing all the marker variables.
    params : Parameters
        Parameters object containing all the simulation parameters.
    xnum : INT
        x-resolution of the simulation domain.
    ynum : INT
        y-resolution of the simulation domain.
    timestep : FLOAT
        Current timestep value.
    xstp_av : FLOAT
        Average grid spacing in x-direction.
    ystp_av : FLOAT
        Average grid spacing in y-direction.
    
    Returns
    -------
    dTn : ARRAY
        Change in temperature due to subgrid diffusion.

    '''
    
    dTn = np.zeros((ynum, xnum))
    
    # also clear the grid.wt ?
    grid.wt = np.zeros((ynum, xnum))
    
    for m in range(0,markers.num):
        # check that the marker is inside the grid
        if (markers.x[m]>= grid.x[0] and markers.x[m]<=grid.x[xnum-1] and markers.y[m]>=grid.y[0] and markers.y[m]<=grid.y[ynum-1]):
            
            # get the grid indexes and marker spacings
            dxm, dym, xn, yn = getMarkerNodeDistances(markers.x[m], markers.y[m], markers.nx[m], markers.ny[m],\
                                                      xnum, ynum, grid, 0)
            # set marker weights
            mwt = 1
            
            # interpolate old nodal temp for marker
            Tm = applyGridContrib(grid.T, xn, yn, dxm, dym)
            # get difference
            dTm = Tm - markers.T[m]
            
            # also get nodal k, rhoCP
            km = applyGridContrib(grid.kT, xn, yn, dxm, dym)
            rhoCPm = applyGridContrib(grid.rhoCP, xn, yn, dxm, dym)
            
            # compute local thermal diff timescale
            Tm_diff = rhoCPm/(km*(2/xstp_av**2 + 2/ystp_av**2))
            
            # compute subgrid diffusion
            s_dif = -params.dsubgridT*timestep/Tm_diff
            if (s_dif < -30):
                s_dif = -30
            dTm = dTm*(1 - np.exp(s_dif))
            
            # correct old temperature for marker
            markers.T[m] += dTm
            
            # interpolate subgrid changes back to nodes
            applyMarkerContrib(dTn, dTm, dxm, dym, xn, yn, mwt)
            applyMarkerContrib(grid.wt, 1, dxm, dym, xn, yn, mwt)
            
    # compute subgrid diffusion for nodes
    for i in range(0,ynum):
        for j in range(0,xnum):
            if (grid.wt[i,j] >= 1e-7):
                dTn[i,j] = dTn[i,j]/grid.wt[i,j]
    
    return dTn

                        

@jit(nopython=True)
def advectMarkers(markers, grid, xnum, ynum, timestep, markmove):
    '''
    Displaces the markers using the calculated grid velocities for the current timestep.

    Parameters
    ----------
    markers : Markers
        Markers object containing all the marker variables, will be updated by this function.
    grid : Grid
        Grid object containing all the grid variables.
    xnum : INT
        x-resolution of the simulation domain.
    ynum : INT
        y-resolution of the simulation domain.
    timestep : FLOAT
        Current timestep value.
    markmove : INT
        Choice of marker advection scheme, 1 = Euler, 4=RK4.

    Returns
    -------
    None.

    '''
    
    # create array to store the sub-position velocities
    vxm = np.zeros((4))
    vym = np.zeros((4))
    espm = np.zeros((4))
    
    for m in range(0,markers.num):
        # check that the marker is inside the grid
        if (markers.x[m]>= grid.x[0] and markers.x[m]<=grid.x[xnum-1] and markers.y[m]>=grid.y[0] and markers.y[m]<=grid.y[ynum-1]):
            
            # save current marker positions
            x_cur = markers.x[m]
            y_cur = markers.y[m]
            
            # loop through number of RK cycles
            for rk in range(0,markmove):
                
                if (rk==0):
                    # first step, use recorded node positions
                    xnmin = markers.nx[m]
                    ynmin = markers.ny[m]
                else:
                    # need to locate the top-left node for the intermediate position
                    xnmin, ynmin = findNearestNode(grid.x, grid.y, xnum, ynum, x_cur, y_cur)
                
                xn = xnmin
                yn = ynmin
                # vx
                if (y_cur>grid.cy[yn+1]):
                    yn = yn+1
                if (yn > ynum-1):
                    yn = ynum-1
                
                # define normalized distance to the upper-left vx node
                dxm = (x_cur - grid.x[xn])/grid.xstp[xn]
                dym = (y_cur - grid.cy[yn])/grid.ystpc[yn]
                
                # calculate marker velocity
                vxm[rk] = applyGridContrib(grid.vx, xn, yn, dxm, dym)
                
                # vy, reset the indicies
                xn = xnmin
                yn = ynmin
                
                if (x_cur>grid.cx[xn+1]):
                    xn = xn+1
                if (xn > xnum-1):
                    xn = xnum-1
                
                # define normalized distance to the upper-left vy node
                dxm = (x_cur - grid.cx[xn])/grid.xstpc[xn]
                dym = (y_cur - grid.y[yn])/grid.ystp[yn]
                
                # calculate marker velocity
                vym[rk] = applyGridContrib(grid.vy, xn, yn, dxm, dym)
                
                
                # now do marker spin, reset indicies
                xn = xnmin
                yn = ynmin
                dxm = (x_cur - grid.x[xn])/grid.xstp[xn]
                dym = (y_cur - grid.y[yn])/grid.ystp[yn]
                
                espm[rk] = applyGridContrib(grid.espin, xn, yn, dxm, dym)
                
                # update coords for next rk point
                if (rk < 3):
                    if (rk < 2):
                        x_cur = markers.x[m] + timestep/2*vxm[rk]
                        y_cur = markers.y[m] + timestep/2*vym[rk]
                    else:
                        x_cur = markers.x[m] + timestep*vxm[rk]
                        y_cur = markers.y[m] + timestep*vym[rk]
        
            # recompute the marker velocity using 4th order RK
            if (markmove==4):
                vxm[0] = (vxm[0] + vxm[1]*2 + vxm[2]*2 + vxm[3])/6
                vym[0] = (vym[0] + vym[1]*2 + vym[2]*2 + vym[3])/6
                espm[0] = (espm[0] + espm[1]*2 + espm[2]*2 + espm[3])/6
        
            # diplace markers according to calculated velocity
            markers.x[m] += timestep*vxm[0]
            markers.y[m] += timestep*vym[0]
        
            # rotate stress on marker according to its spin
            espm[0] = espm[0]*timestep
        
            # store old stresses
            msxx0 = markers.sigmaxx[m]
            msxy0 = markers.sigmaxy[m]
        
            markers.sigmaxy[m] = msxx0*np.sin(2*espm[0]) + msxy0*np.cos(2*espm[0])
            markers.sigmaxx[m] = msxx0*( np.cos(espm[0])**2 - np.sin(espm[0])**2 )\
                                   - msxy0*np.sin(2*espm[0])
        
            # adding marker strain based on grid strain rates
            markers.gII[m] += timestep*np.sqrt(markers.epsxx[m]**2 + markers.epsxy[m]**2)


