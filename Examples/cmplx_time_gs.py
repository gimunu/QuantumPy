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

import quantumPy as qp

# First Create the box
# By default the dimensionality is set to 1 
Radius = 10.0
box = qp.Box(shape = 'Sphere', radius = Radius, spacing = 0.2)
box.write_info() # Write a detailed description of the box

# Create an empty  MeshFunction living on the box
array = np.zeros(box.np)
wf = qp.grid.MeshFunction( array, mesh = box)

# Create a gaussian wave-packet with initial velocity k
X = box.points
k = 4.0
sigma = np.sqrt(2.0)/k
# wf = qp.grid.MeshFunction( np.pi**(-1.0/4.0) * sigma**(-1.0/2.0) *  
#                            np.exp(-X[:]**2.0 / (2.0*sigma**2.0) + 1j*k*X[:]) , mesh = box)

wf[:] = 1.0 + 1j 


def vext(x):
    omega = 1.0
    return 0.5*(x*omega)**2
    
def avext(wf):
    x = wf.mesh.points
    return vext(x)*wf

Vext = qp.Operator(Raction = avext)
Vext.name ='External potential'
Vext.symbol ='Vext'
Vext.formula ='1/2 \omega^2 x^2'

T = qp.system.operators.Kinetic(box)


H = qp.system.operators.Hamiltonian(Operators = [T, Vext])
H.write_info()
print "<H> = %s" % (H.expectationValue(wf))
print "<Vext> = %s" % (Vext.expectationValue(wf))


# Time propagation
# Create the evolution operator
U = qp.td.Propagator(H)
U.write_info()

nt = 500
# dt = 0.005
dt = -1j * 0.001

wft = wf.copy()

anim = 0
if anim:
    pl.ion()
    pl.plot(wft.mesh.points, vext(wft.mesh.points).real)
    line, =pl.plot(wft.mesh.points, (wft.conjugate()*wft).real)
    line.axes.set_xlim(-Radius,Radius) 
    line.axes.set_ylim(0,1) 
    pl.draw()


print "i        t              <E>"
# time-evolution loop
for i in range(0,nt):
    wft = U.apply(wft, dt = dt)         # Apply the propagator
    wft /= np.sqrt((wf*wf.conjugate()).integrate().real)
    E   = U.H.expectationValue(wft)     # Calculate the energy
    v1 = U.H.op_list[0].expectationValue(wft)
    v2 = U.H.op_list[1].expectationValue(wft)
    print "%d\t %1.4f \t%f\t%f\t%f"%(i, i*dt.real, E.real, v1.real, v2.real)
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


