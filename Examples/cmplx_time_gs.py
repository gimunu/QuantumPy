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

#
# Create the simulation box
#
Radius = 5.0
box = qp.Box(shape = 'Sphere', radius = Radius, spacing = 0.05)
box.write_info() # Write a detailed description of the box


# 
# Define the external potential for an harmonic oscillator
#
def vext(x):
    omega = 1.0
    return 0.5*(x*omega)**2
#
# Define the action of the potential on a MeshFuncion.
# In this case is simply a multiplication with the potential
# Action = Vext * wf
#    
def avext(wf):
    x = wf.mesh.points
    return vext(x)*wf

#
# Allocate and initialize the External potential operator.
#
Vext = qp.Operator()
Vext.action(avext)      # Set operator's action 
Vext.name ='External potential'
Vext.symbol ='Vext'
Vext.formula ='1/2 \omega^2 x^2'

#
# Kinetic operator
#
# T = qp.system.operators.Kinetic(box, Strategy = 'fs', Bc = 'periodic')
# T = qp.system.operators.Kinetic(box, Strategy = 'fd', Order = 1, Bc = 'periodic')
T = qp.Kinetic(box, Strategy = 'fd', Order = 3, Bc = 'zero')

#
# The Hamiltonian
#
H = qp.Hamiltonian(Operators = [T, Vext])


#
# Imaginary-time propagation
# Create the evolution operator
U = qp.td.Propagator(H)
U.write_info()

# imaginary time-step
dt = -1j * 1 

# Enetgy threshold
Eth = 1e-6

# Create an empty  MeshFunction 
array = np.zeros(box.np, dtype=complex)
wft = qp.grid.MeshFunction( array, mesh = box)

wft[:] = np.random.rand(box.np)  # The initial guess of the 

anim = 1
if anim:
    pl.ion()
    pl.plot(wft.mesh.points, vext(wft.mesh.points).real)
    line, =pl.plot(wft.mesh.points, (wft.conjugate()*wft).real, marker='o')
    line.axes.set_xlim(-Radius,Radius) 
    line.axes.set_ylim(-0.01,1.0) 
    pl.draw()


print "i        t              <E>"
# time-evolution loop
Eold = 0.0
for i in range(0, 100):
    # Apply the propagator
    wft = U.apply(wft, dt = dt)         
    # Normalize 
    norm2 =  (wft*wft.conjugate()).integrate().real
    wft /= np.sqrt(norm2)
    # Calculate the energy
    E   = U.H.expectationValue(wft)
    dE  = np.abs(Eold - E)/np.abs(E)
    Eold = E
    if dE < Eth: 
        exit      
    # <T>
    ET = U.H.op_list[0].expectationValue(wft) 
    # <Vext>
    EV = U.H.op_list[1].expectationValue(wft) 
    print "%d\t %1.4f \t%f\t%f\t%f\t%f"%(i, i*dt.real, E.real, ET.real, EV.real, dE)
    if anim:
        line.set_ydata((wft.conjugate()*wft).real)
        pl.draw()

print "Converged."
    

if 0:
    pl.rc('font', family='serif')
    pl.rc('font', size=12)
    pl.rc('legend', fontsize=10)

    fig = pl.figure()
    p1 = fig.add_subplot(1, 2, 1, title = 'Wavefunction - RS')
    p1.plot(wf.mesh.points, wf.real,  lw=1, color='r', label='$wf.real$')
    p1.plot(wf.mesh.points, wf.imag,  lw=1, color='b', label='$wf.imag$')
    p1.plot(wf.mesh.points, (wf.conjugate()*wf).real,  lw=2, color='g', label='$den$')
    p1.plot(wf.mesh.points, (wft.conjugate()*wft).real,  lw=2, color='black', label='$den(t)$')
    # p1.plot(Lwf.mesh.points, (Lwf.conjugate()*Lwf).real,  lw=2, color='g', label='$density$')
    p1.legend()

    p2 = fig.add_subplot(1, 2, 2, title = 'Wavefunction - FS')
    p2.plot(Fwf.mesh.FSpoints, Fwf.real,  lw=1, color='r', label='$Fwf.real$')
    p2.plot(Fwf.mesh.FSpoints, Fwf.imag,  lw=1, color='b', label='$Fwf.imag$')
    p2.plot(Fwf.mesh.FSpoints, (Fwf.conjugate()*Fwf).real,  lw=2, color='g', label='$density$')
    # p2.plot(Fwf.mesh.FSpoints, Fwf.mesh.FSpoints,  lw=2, marker='o', color='g', label='$density$')
    p2.legend()

    pl.show()   


