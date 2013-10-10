#!/usr/bin/env python

# Copyright (C) 2013 U. De Giovannini
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
# 02110-1301, USA.

from __future__ import division

import numpy as np
import pylab as pl
import scipy as sp
import pylab as pl



import sys
sys.path.append('../')

import quantumPy as qp
import time as TTtime

def gaussian_wp(mesh, sigma, k):
    X = mesh.points
    wf = qp.MeshFunction( np.pi**(-1.0/4.0) * sigma**(-1.0/2.0) *  
                               np.exp(-X[:]**2.0 / (2.0*sigma**2.0) + 1j*k*X[:]) , mesh = mesh)
    return wf   

def gaussian_wpT(mesh, sigma, k, t):
    X = mesh.points
    wf = qp.MeshFunction( np.exp((1j*X[:]**2 + k*(-k*t +2*X[:])*sigma**2)/(2*(t-1j*sigma**2)))*  
                               np.sqrt(sigma)/(np.pi**(1./4.)*np.sqrt(1j*t+sigma**2)) 
                               , mesh = mesh)
    return wf 

def mask (widht, RB = None, type = 'sin2'):    

    if  type == 'sin2':
        def func(x, wd = widht):
            y = x.copy()
            y[:] = 0.0 
            R = np.abs(x.max()) if not RB else RB
            for i, value in np.ndenumerate(x):
                dd = np.abs(np.abs(x[i])-R) 
                if dd < wd:
                    y[i] = np.sin(dd/wd * np.pi/2)**2
                else:
                    y[i] = 1.0
                # else:
                #     y[i] = 0.0
            return y 
    elif type == 'unit_box':
        def func(x, wd = widht):
            y = x.copy()
            y[:] = 0.0 
            R = np.abs(x.max()) if not RB else RB
            for i, value in np.ndenumerate(x):
                dd = np.abs(np.abs(x[i])-R) 
                if dd < wd:
                    y[i] = 0.0
                else:
                    y[i] = 1.0
            return y 
         
                
    return func

def Impotf (widht, height, type = 'sin2'):
    
    if type == 'sin2':    
        def sin2(x, wd = widht, h = height):
            y = x.copy()
            y[:] = 0.0 
            R = np.abs(x.max())
            for i, value in np.ndenumerate(x):
                dd = np.abs(np.abs(x[i])-R) 
                if dd < wd:
                    y[i] = 1 - np.sin(dd/wd * np.pi/2)**2
                else:
                    y[i] = 0.0
                # else:
                #     y[i] = 0.0
            return h *y 
                
        return sin2
    elif type == 'poly':
        pass


# def ses_f(g_func, theta):
    

def evolve_mask(ABWidth, k, type, verbose = True, anim = False, quick = False):

    dR = 0.1
    dt = 0.01
    sigma = np.sqrt(2.0)/abs(k)
    # dR = 0.25*sigma
    # dt = 0.005/(k**2/2) 
    Radius =  3*sigma + ABWidth 
    # Radius =  5 
    T =  2.0 * Radius/abs(k)

    box = qp.Box(shape = 'Sphere', radius = Radius, spacing = dR)

    if verbose:
        box.write_info() # Write a detailed description of the box


    maskf  = None
    impotM = None
    maskM = None

    # H = qp.hamiltonian(box, Strategy = 'fs', Bc = 'periodic')
    H = qp.hamiltonian(box) 
    if type == 'cap':
        eta = 0.2
        impotf = Impotf(ABWidth, height = -1j * eta)
        impotM = qp.MeshFunction(impotf(box.points), box)
        Vcap = qp.scalar_pot(impotM, box)
        H += Vcap  
    if type == 'cap_mask':
        maskf = mask(ABWidth, RB = Radius + 0.*dR)
        maskM = qp.MeshFunction(maskf(box.points), box)
        Mm = qp.MeshFunction(maskf(box.points), box)
        Mm = np.log(Mm)
        # Mm[-1] = Mm[-2]
        # Mm[0]  = Mm[1]
        GO = qp.Gradient(box)
        LO = qp.Laplace(box)
        GMm = GO.apply(Mm)
        LMm = LO.apply(Mm)
        # print Mm
        # print GMm
        # print LMm
        # print box.points
        impotM =  1j/dt * Mm  - 1./4.*LMm - 1./12. * GMm**2  
        # pl.plot(box.points, impotM.real, lw = 2, label='Re[Impot]')    
        # pl.plot(box.points, impotM.imag, lw = 2, label='Im[Impot]')
        # # pl.legend()
        # pl.show()   
        # exit()
        Vstatic = qp.scalar_pot(impotM)

        VLMm = qp.scalar_pot(-1./2.*GMm) 
        Vcap = Vstatic + VLMm * GO 
        H += Vcap  

        

    if type == 'cap_ses':
        pass
        
    U = qp.td.propagator(H, method = 'etrs', exp_order = 4)

    if type == 'cap_mask':
        # force the wavefuncion at the edges to be well beaved
        # by setting to zero the stencil points at the boundaries
        Ubox = qp.scalar_pot(mask(12*dR, type = 'unit_box'), box)
        U *= Ubox

    if type == 'mask':
        maskf = mask(ABWidth)
        maskM = qp.MeshFunction(maskf(box.points), box)
        M = qp.scalar_pot(maskf)
        U = U*M

    if type == 'mask_cap':
        eta = 0.2*1000
        impotf = Impotf(ABWidth, height = -1j * eta)
        impotM = qp.MeshFunction(impotf(box.points), box)
        maskM = np.exp(-1j * dt * impotM)
        M = qp.scalar_pot(maskM)
        U = U*M

    if verbose:
        if 'Vcap' in locals():
            Vcap.write_info()
        H.write_info()
        U.write_info()

    wft   = gaussian_wp(box, sigma, k)
    wftEx = gaussian_wpT(box, sigma, k, 0.0)

    if anim and not quick:
        pl.ion()
        if maskM != None:
            pl.plot(wft.mesh.points, maskM.real, lw = 2, label='Mask')
            pl.plot(wft.mesh.points, maskM.imag, lw = 2, label='Im[Mask]')
        if impotM != None:            
            pl.plot(wft.mesh.points, abs(impotM.real), lw = 2, label='Re[Impot]')    
            pl.plot(wft.mesh.points, abs(impotM.imag), lw = 2, label='Im[Impot]')
        line, =pl.plot(wft.mesh.points, (wft.conjugate()*wft).real, color = 'G', marker='o', label='|wf0|^2')
        line.axes.set_xlim(-Radius,Radius) 
        line.axes.set_ylim(-0.01,1.0) 
        lineEx, =pl.plot(wftEx.mesh.points, (wftEx.conjugate()*wftEx).real, color = 'R', label='|wfexact|^2')
        pl.legend()
        pl.draw()

    if verbose:
        print "i        t              <E>"
    # time-evolution loop

    Niter = int(T/dt)
    t1 = TTtime.time()
    for i in range(0, Niter):
        T = i*dt
        wft = U.apply(wft, dt = dt) 
        if not quick:
            N = (wft.conjugate()*wft).integrate().real
            E   = H.expectationValue(wft)
            wftEx = gaussian_wpT(box, sigma, k, T)
            NEx = (wftEx.conjugate()*wftEx).integrate().real
            if anim:
                line.set_ydata((wft.conjugate()*wft).real)
                lineEx.set_ydata((wftEx.conjugate()*wftEx).real)
                pl.draw()
            t2 = TTtime.time() 
            DT = t2 - t1
            t1 = t2   
            if verbose :
                print "%d\t %s \t%f \t%f \t%f \t%2.3f"%(i, i*dt, E.real, N, NEx, DT)
                
            
    

    # if anim:
    #     pl.ioff()
    #     pl.show()   

    N = (wft.conjugate()*wft).integrate().real
    wftEx = gaussian_wpT(box, sigma, k, T)
    NEx = (wftEx.conjugate()*wftEx).integrate().real
    
    return (N, NEx)

#############    
# MAIN 
############

N, Nex = evolve_mask(10., k =  1. , type = 'mask_cap', quick = False, verbose = True, anim = True)

print N, Nex
