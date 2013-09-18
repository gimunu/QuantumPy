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
Radius = 4.0
box = qp.Box(shape = 'Sphere', radius = Radius, spacing = 0.2)
box.write_info() # Write a detailed description of the box


# 
# Define the external potential for an harmonic oscillator
#
def vext(x):
    omega = 1.0
    # return 0.0*x
    return 0.5*(x*omega)**2
#
# Allocate and initialize the External potential operator.
#
Vext = qp.scalar_pot(vext)
#
# Kinetic operator
#
# T = qp.system.operators.Kinetic(box, Strategy = 'fs', Bc = 'periodic')
# T = qp.system.operators.Kinetic(box, Strategy = 'fd', Order = 1, Bc = 'periodic')
T = qp.Kinetic(box, Strategy = 'fd', Order = 1, Bc = 'periodic')
#
# The Hamiltonian
#
# H = qp.Hamiltonian(Operators = [T, Vext])

H = T + Vext + Vext
H.write_info()
print H.op_list
exit()

#################################################
# Imaginary-time propagation
# Create the evolution operator
U = qp.td.Propagator(H)

# U = qp.exponential(H)
U.write_info()

dtau = -1j * 0.007 
Eth = 1e-6

# Create an empty  MeshFunction 
array = np.zeros(box.np, dtype=complex)
wft = qp.grid.MeshFunction( array, mesh = box)

wft = qp.grid.MeshFunction( np.random.rand(box.np)+0.j, mesh = box)


anim = 1
if anim:
    pl.ion()
    pl.plot(wft.mesh.points, vext(wft.mesh.points).real, lw = 2, marker='o', label='Vext')
    line, =pl.plot(wft.mesh.points, (wft.conjugate()*wft).real, marker='o', label='|wf0|^2')
    line.axes.set_xlim(-Radius,Radius) 
    line.axes.set_ylim(-0.01,1.0) 
    pl.legend()
    pl.draw()


print "i        t              <E>"
# time-evolution loop
Eold = 0.0
for i in range(0, int(1e3)):
    # Apply the propagator
    wft = U.apply(wft, dt = dtau)         
    # wft = U.apply(wft, exp_step = -1j * dtau)         
    # Normalize 
    norm2 =  (wft*wft.conjugate()).integrate().real
    wft /= np.sqrt(norm2)
    # Calculate the energy
    E   = H.expectationValue(wft)
    dE  = np.abs(Eold - E)/np.abs(E)
    Eold = E
    # <T>
    ET = T.expectationValue(wft) 
    # <Vext>
    EV = Vext.expectationValue(wft) 
    print "%d\t %1.4f \t%f\t%f\t%f\t%1.4e"%(i, i*dtau.real, E.real, ET.real, EV.real, dE)
    if anim:
        line.set_ydata((wft.conjugate()*wft).real)
        pl.draw()

    if (dE < Eth):
        print "Converged!"
        break      

print "Finished."

if anim:
    pl.ioff()
    pl.show()   
    

