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


import numpy as np
import pylab as pl
import scipy as sp
import pylab as pl

import sys
sys.path.append('../')

import quantumPy as qp

#
# First Create the box
# We are working in one-dimension by default
#
Radius = 5.0
box = qp.Box(shape = 'Sphere', radius = Radius, spacing = 0.1)
box.write_info() # Write a detailed description of the box

#
# Create an empty  MeshFunction
#
array = np.zeros(box.np)
wf = qp.grid.MeshFunction( array, mesh = box)

#
# Create a gaussian wave-packet with initial velocity k
#
X = box.points
k = 4.0
sigma = np.sqrt(2.0)/k
wf = qp.grid.MeshFunction( np.pi**(-1.0/4.0) * sigma**(-1.0/2.0) *  
                           np.exp(-X[:]**2.0 / (2.0*sigma**2.0) + 1j*k*X[:]) , mesh = box)
#
# Check the normalization
#
# print "Norm = %f" % (wf*wf.conjugate()).integrate().real  

#
# Create the gradient operator \nabla
# The expectation value on our gaussian packet should be
# it's momentum k: <- i \nabla> = <P>  
#
Grad = qp.system.Gradient(box, Strategy = 'fd')
# Grad.write_info()
print "* <P> = %12.6f" % ( (- 1j * Grad.expectationValue(wf)).real)

#
# Create the kinetic operator T = -1/2 \nabla^2 
# The expectation value should be equal to:
#  <T> = 5/4 <P>^2/2 (= 10 for our gaussian with k = 4 )
#
# T = qp.system.operators.Kinetic(box, Strategy = 'fd', Order = 1, Bc = 'periodic')
T = qp.system.operators.Kinetic(box, Strategy = 'fd', Order = 4, Bc = 'zero')
print "* <T> = %12.6f" % (T.expectationValue(wf).real)



#
# Add the kinetic operator to the list of operators 
# composing the Hamiltonian.  
#
H = qp.system.operators.Hamiltonian(Operators = [T])
H.write_info()
print "* <H> = %12.6f" % (H.expectationValue(wf).real)

#
# Calculate kinetic energy in Fourier-space
# to check the FS helpers
#
Fwf = qp.base.math.rs_to_fs(wf)
tmp =Fwf.mesh.FSpoints**2.0 * Fwf
Lwf = qp.base.math.fs_to_rs(tmp, return_mf = True)
E = 0.5*(wf.conjugate()*Lwf).integrate()
print "Ek = %s"%(E)

#
# Test operator composition
#
OP = qp.system.Operator(Operators = [T,T,Grad])
OP.write_info()



#
# Time propagation
# Create the evolution operator
#
U = qp.td.Propagator(H)
U.write_info()

nt = 500
dt = 0.001

wft = wf.copy()

#
# Plot the evolution of the wavepacket?
#
anim = 1
if anim:
    pl.ion()
    line, =pl.plot(wft.mesh.points, (wft.conjugate()*wft).real)
    line.axes.set_xlim(-Radius,Radius) 
    pl.draw()


print "i        t              <E>"
# time-evolution loop
for i in range(0,nt):
    wft = U.apply(wft, dt = dt)         # Apply the propagator
    E   = U.H.expectationValue(wft)     # Calculate the energy
    print "%d\t %1.4f \t%12.6f"%(i, i*dt.real, E.real)
    if anim:
        line.set_ydata((wft.conjugate()*wft).real)
        pl.draw()

    

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


