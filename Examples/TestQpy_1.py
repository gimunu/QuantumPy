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

import quantumPy as qp
from quantumPy import *

# First Create the box
# By default the dimensionality is set to 1 
box = qp.grid.Box(shape = 'Sphere', radius = 10.0, spacing = 0.1)
box.write_info() # Write a detailed description of the box

# Create an empty  MeshFunction living on the box
array = np.zeros(box.np)
mf = qp.grid.MeshFunction( array, mesh = box)

# Create a gaussian wave-packet with initial velocity k
X = box.points
k = 4.0
sigma = np.sqrt(2.0)/k
mf = qp.grid.MeshFunction( np.pi**(-1.0/4.0) * sigma**(-1.0/2.0) *  np.exp(-X[:]**2.0 / (2.0*sigma**2.0) + 1j*k*X[:]) , mesh = box)
# Check the normalization
# print "Norm = %f" % (mf*mf.conjugate()).integrate().real  


# Create a Laplace operator
Lap = qp.system.Laplacian(box, Strategy = 'Fourier')

# Create the kinetic operator 
# T = qp.system.Kinetic(box, Strategy = 'Fourier')
T = qp.system.operators.Kinetic(box)

Ops = [T]

HBaseOp = qp.system.Hamiltonian(Operators = Ops)
HBaseOp.write_info()

print "<H> = %s" % (HBaseOp.expectationValue(mf))
print "<T> = %s" % (T.expectationValue(mf))

I = qp.system.operators.Identity()
print "<I> = %s" % (I.expectationValue(mf))

# HBaseOp.apply(mf)

Fmf = qp.base.math.rs_to_fs(mf)
tmp =Fmf.mesh.FSpoints**2.0 * Fmf
Lmf = qp.base.math.fs_to_rs(tmp, return_mf = True)
E = 0.5*(mf.conjugate()*Lmf).integrate()
print "E = %s"%(E)

# ET = (mf.conjugate()*T.apply(mf)).integrate()
ET = T.expectationValue(mf)
print"ET = %s"%(ET)

# Test operator composition
# OP = qp.system.Operator(Operators = [T,T,Lap])
# OP.write_info()


# Time propagation
# Create the evolution operator
U = qp.td.Propagator(HBaseOp)
U.write_info()

nt = 400
dt = 0.001


wft = mf.copy()
print "i        t              <E>"
# time-evolution loop
for i in range(0,nt):
    wft = U.apply(wft, dt = dt)         # Apply the propagator
    E   = U.H.expectationValue(wft)     # Calculate the energy
    print "%d\t %1.4f \t%f"%(i, i*dt, E.real)
    


if True:
    pl.rc('font', family='serif')
    pl.rc('font', size=12)
    pl.rc('legend', fontsize=10)

    fig = pl.figure()
    p1 = fig.add_subplot(1, 2, 1, title = 'Wavefunction - RS')
    p1.plot(mf.mesh.points, mf.real,  lw=1, color='r', label='$mf.real$')
    p1.plot(mf.mesh.points, mf.imag,  lw=1, color='b', label='$mf.imag$')
    p1.plot(mf.mesh.points, (mf.conjugate()*mf).real,  lw=2, color='g', label='$den$')
    p1.plot(mf.mesh.points, (wft.conjugate()*wft).real,  lw=2, color='black', label='$den(t)$')
    # p1.plot(Lmf.mesh.points, (Lmf.conjugate()*Lmf).real,  lw=2, color='g', label='$density$')
    p1.legend()

    p2 = fig.add_subplot(1, 2, 2, title = 'Wavefunction - FS')
    p2.plot(Fmf.mesh.FSpoints, Fmf.real,  lw=1, color='r', label='$Fmf.real$')
    p2.plot(Fmf.mesh.FSpoints, Fmf.imag,  lw=1, color='b', label='$Fmf.imag$')
    p2.plot(Fmf.mesh.FSpoints, (Fmf.conjugate()*Fmf).real,  lw=2, color='g', label='$density$')
    # p2.plot(Fmf.mesh.FSpoints, Fmf.mesh.FSpoints,  lw=2, marker='o', color='g', label='$density$')
    p2.legend()

    pl.show()   
