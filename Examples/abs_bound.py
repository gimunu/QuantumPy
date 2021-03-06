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

def Impotf (widht, height, type = 'sin2', order = 3, **kwds):
    
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
            return h *y 
                
        return sin2
    elif type == 'poly':
        def poly(x, wd = widht, h = height, order = order):
            y = x.copy()
            y[:] = 0.0 
            R = np.abs(x.max())
            for i, value in np.ndenumerate(x):
                dd = np.abs(np.abs(x[i])-R) 
                if dd < wd:
                    y[i] = np.abs(dd-wd)**order
                else:
                    y[i] = 0.0
            return h * y 
                
        return poly

    elif type == 'cosh':
        alpha = kwds.pop('alpha', 3.* 2./widht)
        def cosh(x, wd = widht, h = height, alpha = alpha):
            y = x.copy()
            y[:] = 0.0 
            R = np.abs(x.max())
            for i, value in np.ndenumerate(x):
                dd = np.abs(np.abs(x[i])-R) 
                if dd < wd:
                    y[i] = 1/np.cosh(alpha * dd)**2
                else:
                    y[i] = 0.0
            return h * y 
                
        return cosh

# Smooth exterior complex scaling
def ses_f(g_func, theta):
    def f(x, x0, theta = theta):
        y = x.copy()
        y[:] = 0.0
        y = 1.0 + (np.exp(1j*theta) - 1) * g_func(x, x0)
        return y 
    return f
    
def ses_g(type = 'tanh', **kwds):
    
    if type == 'tanh':  
        lam = kwds.get('lam', 1)    
        def g(x, x0, lam = lam):
            y = x.copy()
            y[:] = 0.0
            y = 1. + 0.5 * (np.tanh(lam*(x - x0)) - np.tanh(lam*(x + x0)))
            return y
    elif type == 'sin2':
        def g(x, x0):
            y = x.copy()
            y[:] = 0.0 
            R = np.abs(x.max())
            wd = abs(R - x0)
            for i, value in np.ndenumerate(x):
                dd = np.abs(np.abs(x[i])-R) 
                if dd < wd:
                    y[i] = 1 - np.sin(dd/wd * np.pi/2)**2
                else:
                    y[i] = 0.0
            return y 
            
                
    return g

def evolve_mask(ABWidth, k, type, verbose = True, anim = False, quick = False, **kwds):

    dR =  kwds.get('dR', 0.1) # 0.1
    dt =  kwds.get('dt', 0.01)
    sigma = 4.0 * np.sqrt(2.0)/abs(k)
    # dR = 0.25*sigma
    # dt = 0.005/(k**2/2) 
    Radius =  3*sigma + ABWidth 
    # Radius =  5 
    T =  2. * Radius/abs(k)

    # qp.base.messages.DEBUG_LEVEL = qp.base.messages.DEBUG_STACKTRACE + 2
    # qp.base.messages.DEBUG_LEVEL = 2

    # box = qp.box(shape = 'Sphere', radius = Radius, spacing = dR, dim = 2)
    box = qp.box(shape = 'Sphere', radius = Radius, spacing = dR, dim = 1)
            
    if verbose:
        box.write_info() # Write a detailed description of the box


    maskf  = None
    impotM = None
    maskM = None


    # The initial wavepackets
    wft   = gaussian_wp(box, sigma, k)
    wftEx = gaussian_wpT(box, sigma, k, 0.0)


    # H = qp.hamiltonian(box, Strategy = 'fs', Bc = 'periodic') 
    H = qp.hamiltonian(box, Strategy = 'fd', Bc = 'zero', Order = 4 ) 
    # H = qp.hamiltonian(box)

    if type == 'cap_sin2':
        eta = kwds.get('eta', 0.2)
        impotf = Impotf(ABWidth, height = -1j * eta)
        impotM = qp.MeshFunction(impotf(box.points), box)
        Vcap = qp.scalar_pot(impotM, box)
        H += Vcap  

    if type == 'cap_poly':
        eta = kwds.get('eta', 0.03)
        impotf = Impotf(ABWidth, height = -1j * eta, type = 'poly', order = kwds.get('order', 3))
        impotM = qp.MeshFunction(impotf(box.points), box)
        Vcap = qp.scalar_pot(impotM, box)
        H += Vcap  

    if type == 'cap_cosh':
        NN = 6.
        alpha = kwds.get('alpha', NN* 2./ABWidth)
        n = 1
        U = kwds.get('U', (alpha**2/8) * ((2*n + 1)**2 + 1))
        impotf = Impotf(ABWidth, height = -1j * U, type = 'cosh', alpha = alpha)
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
        if kwds.get('LeadingOrderOnly', False):
            impotM =  1j/dt * Mm  
        else:             
            impotM =  1j/dt * Mm  - 1./2.*LMm + 1./6. * GMm**2  
        # pl.plot(box.points, impotM.real, lw = 2, label='Re[Impot]')    
        # pl.plot(box.points, impotM.imag, lw = 2, label='Im[Impot]')
        # # pl.legend()
        # pl.show()   
        # exit()
        Vstatic = qp.scalar_pot(impotM)

        if kwds.get('LeadingOrderOnly', False):
            Vcap = Vstatic  
        else:
            VLMm = qp.scalar_pot(-GMm) 
            Vcap = Vstatic + VLMm * GO 
        H += Vcap  

        

    if type == 'cap_ses':
        theta  = kwds.get('theta', 0.1)
        lam = kwds.get('lam', 1.0)
        x0 = Radius - ABWidth
        g = ses_g(type = kwds.get('gfunc','tanh'), lam = lam)
        f = ses_f(g, theta )
        fM = qp.MeshFunction(f(box.points, x0), box)
        maskM = fM
        GO = qp.Gradient(box)
        LO = qp.Laplace(box)
        DfM  = GO.apply(fM)
        DDfM = LO.apply(fM)
        V0 = DDfM/fM**3. - (5./8.) * DfM**2./fM**4.
        V1 = DfM / fM**3.
        V2 = 0.5*(1. - fM**(-2.))
        Vcap = qp.scalar_pot(V0) + qp.scalar_pot(V1) * GO + qp.scalar_pot(V2) * LO
        # Vcap =  V1 * GO + LO * V2  + V0
        H += Vcap
        
        impotM = Vcap.apply(wft)
        # pl.plot(box.points, fM, lw = 2) 
        # pl.plot(box.points, V0, lw = 2, label = 'V0') 
        # pl.plot(box.points, V1, lw = 2, label = 'V1') 
        # pl.plot(box.points, V2, lw = 2, label = 'V2')
        # pl.legend() 
        # pl.show()   
        # exit()
        
        
    U = qp.td.propagator(H, method = 'etrs', exp_order =  kwds.get('exp_order', 4))

    if type == 'cap_poly' :
        # force the wavefuncion at the edges to be well beaved
        # by setting to zero the stencil points at the boundaries
        Ubox = qp.scalar_pot(mask(12*dR, type = 'unit_box'), box)
        U = Ubox *U

    if type == 'cap_mask' or type == 'cap_ses':
        # force the wavefuncion at the edges to be well beaved
        # by setting to zero the stencil points at the boundaries
        Ubox = qp.scalar_pot(mask(12*dR, type = 'unit_box'), box)
        U = Ubox*U

    if type == 'mask_sin2':
        maskf = mask(ABWidth)
        maskM = qp.MeshFunction(maskf(box.points), box)
        M = qp.scalar_pot(maskf)
        U = M*U

    if type == 'mask_cap_sin2':
        eta = kwds.get('eta', 0.2)
        impotf = Impotf(ABWidth, height = -1j * eta)
        impotM = qp.MeshFunction(impotf(box.points), box)
        maskM = np.exp(-1j * dt * impotM)
        M = qp.scalar_pot(maskM)
        U = M*U

    if type == 'mask_cap_poly':
        eta = kwds.get('eta', 0.03)
        impotf = Impotf(ABWidth, height = -1j * eta, type = 'poly', order = 3)
        impotM = qp.MeshFunction(impotf(box.points), box)
        maskM = np.exp(-1j * dt * impotM)
        M = qp.scalar_pot(maskM)
        U = M*U

    if type == 'mask_cap_ses':
        theta  = kwds.get('theta', 0.1)
        lam = kwds.get('lam', 1.0)
        x0 = Radius - ABWidth
        g = ses_g(type = 'tanh', lam = lam)
        f = ses_f(g, theta = 0.2)
        fM = qp.MeshFunction(f(box.points, x0), box)
        GO = qp.Gradient(box)
        LO = qp.Laplace(box)
        DfM  = GO.apply(fM)
        DDfM = LO.apply(fM)
        V0 = DDfM/fM**3. - (5./8.) * DfM**2./fM**4.
        maskM = np.exp(-1j * dt * V0)
        V1 = DfM / fM**3.
        V2 = 0.5*(1. - fM**(-2.))
        Vcap = qp.scalar_pot(V0) + qp.scalar_pot(V1) * GO + qp.scalar_pot(V2) * LO
        M = qp.exponential(Vcap, exp_step = -1j*dt, order =  kwds.get('mask_exp_order', 4)) 
        U = M*U
        #force borders to be zero 
        Ubox = qp.scalar_pot(mask(12*dR, type = 'unit_box'), box)
        U = Ubox*U
        

    if verbose:
        if 'Vcap' in locals():
            Vcap.write_info()
        H.write_info()
        U.write_info()


    if anim and not quick:
        pl.ion()
        if maskM != None:
            # pl.plot(wft.mesh.points, maskM.real, lw = 2, label='Mask')
            # pl.plot(wft.mesh.points, maskM.imag, lw = 2, label='Im[Mask]')
            lineMRE, = qp.plot(maskM.real, lw = 2, label='Mask')
            lineMIM, = qp.plot(maskM.imag, lw = 2, label='Im[Mask]')
        if impotM != None:            
            # pl.plot(wft.mesh.points, abs(impotM.real), lw = 2, label='Re[Impot]')    
            # pl.plot(wft.mesh.points, abs(impotM.imag), lw = 2, label='Im[Impot]')
            lineCAPRE, = qp.plot(abs(impotM.real), lw = 2, label='Re[Impot]')    
            lineCAPIM, = qp.plot(abs(impotM.imag), lw = 2, label='Im[Impot]')
        # line, =pl.plot(wft.mesh.points, (wft.conjugate()*wft).real, color = 'G', marker='o', label='|wf0|^2')
        line, = qp.plot((wft.conjugate()*wft).real, color = 'G', marker='o', label='|wf0|^2')
        line.axes.set_xlim(-Radius,Radius) 
        if   type == 'cap_ses':
            line.axes.set_ylim(-0.2,0.5)
        elif type == 'mask_cap_ses':
            line.axes.set_ylim(-0.1,1.2)
        else:    
            line.axes.set_ylim(-0.01,1.0)
        
        pl.axvline(x = Radius - ABWidth ,    color='black', ls = '--')
        pl.axvline(x = - (Radius - ABWidth), color='black', ls = '--' )
             
        # lineEx, =pl.plot(wftEx.mesh.points, (wftEx.conjugate()*wftEx).real, color = 'R', label='|wfexact|^2')
        lineEx, =qp.plot((wftEx.conjugate()*wftEx).real, color = 'R', label='|wfexact|^2')
        pl.legend( loc='upper left')
        pl.draw()

    if verbose:
        print "i        t              <E>"
    # time-evolution loop

    NFiles=kwds.get('saveNumber', 10)

    

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
                if type == 'mask_cap_ses':                
                    maskM = np.exp(-1j*dt * Vcap.apply(wft))
                    lineMRE.set_ydata(maskM.real)
                    lineMIM.set_ydata(maskM.imag)
                    
                if type == 'cap_ses':                
                    impotM = Vcap.apply(wft)
                    lineCAPRE.set_ydata(impotM.real)
                    lineCAPIM.set_ydata(impotM.imag)
                    
                line.set_ydata((wft.conjugate()*wft).real)
                lineEx.set_ydata((wftEx.conjugate()*wftEx).real)
                pl.draw()
            t2 = TTtime.time() 
            DT = t2 - t1
            t1 = t2  
            
            if i%int(Niter/NFiles) == 0 :
                if   'wfs' in kwds.get('save'): 
                    qp.write(wft,"wft_%06d"%i)
                if 'impot' in kwds.get('save'): 
                    qp.write(impotM,"impot_%06d"%i)
                # elif 'mask' in kwds.get('save')
             
            if verbose :
                print "%d\t %s \t%f \t%f \t%f \t(%2.3f s) -- %e"%(i, i*dt, E.real, N, NEx, DT, wft[-1].real)
                
            
    

    # if anim:
    #     pl.ioff()
    #     pl.show()   

    N = (wft.conjugate()*wft).integrate().real
    wftEx = gaussian_wpT(box, sigma, k, T)
    NEx = (wftEx.conjugate()*wftEx).integrate().real
    
    RA = Radius - ABWidth
    # print RA
    def segment(pos):
        return qp.segment(pos, - RA, RA)
    intRegion = qp.submesh(segment, box)
    # print intRegion.pindex
    # print box.points[intRegion.pindex]
    
    NAEx = (wftEx.conjugate()*wftEx).integrate(intRegion).real
    NA = (wft.conjugate()*wft).integrate(intRegion).real
    
    wfdiff = wft - wftEx
    diff = (wfdiff.conjugate()*wfdiff).integrate(intRegion).real
    diff = wfdiff.norm2(intRegion).real
    
    # # calculate the reflection coefficients with ffts
    # print intRegion.points
    # print wft[intRegion.pindex]
    # cube = qp.Cube(intRegion, Attributes = 'RS + FS') 
    # cwft = qp.mesh_to_cube(wft[intRegion.pindex], cube)
    # cwfEx0 = qp.mesh_to_cube(gaussian_wpT(box, sigma, k, 0.0)[intRegion.pindex], cube)
    # FcwfEx0 = qp.base.math.rs_to_fs(cwfEx0)
    # Fcwft = qp.base.math.rs_to_fs(cwft)
    # 
    # pl.figure(0)
    # pl.plot(cube.FSpoints, np.abs(FcwfEx0)**2, lw = 2, label='FcwfEx0')
    # pl.plot(cube.FSpoints, np.abs(Fcwft)**2, lw = 2, label='Fcwft')
    # pl.plot(cube.FSpoints, np.abs(Fcwft)**2/np.abs(FcwfEx0)**2, lw = 2, marker='o',label='R')
    # # pl.xscale('log')
    # # pl.xlim([0,1e6])
    # pl.legend()
    # pl.ioff()
    # pl.show()
    
    return (N, NEx, NA, NAEx, diff)

#############    
# MAIN 
############
if __name__ == '__main__':

    N, Nex, NA, NAex, diff = evolve_mask(10, k =  20 , type = 'cap_sin2', eta = 0.01,  
                                         quick = False, verbose = True, anim = True)

    print N, Nex, NA, NAex, diff
