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

# Python standard libs
import sys

# Module to test
import quantumPy as qp
from quantumPy import *


box = qp.grid.Box(shape = 'Sphere', radius = 10.0, spacing = 0.1)
box.write_info()
# print "****"
# mesh = qp.grid.Mesh()
# mesh.write_info()

array = np.zeros(box.np)
mf = qp.grid.MeshFunction( array, mesh = box)
# print "before "+ mf.__class__.__name__

X = box.points
k = 4.0
sigma = np.sqrt(2.0)/k
mf = qp.grid.MeshFunction( np.pi**(-1.0/4.0) * sigma**(-1.0/2.0) *  np.exp(-X[:]**2.0 / (2.0*sigma**2.0) + 1j*k*X[:]) , mesh = box)
# print "norm = %s" % (mf*mf.conjugate()).integrate()  
# mf /= np.sqrt((mf*mf.conjugate()).integrate()) 
# print "after "+ mf.__class__.__name__

mf2 = mf.copy()
mf2[:] = 0.0
# print "mf2  "+ mf2.__class__.__name__
# print mf
# print X

Lap = qp.system.Laplacian(box, Strategy = 'Fourier')

T = qp.system.Kinetic(box, Strategy = 'Fourier')
# T = qp.system.operators.Kinetic(box)

Ops = [T]

HBaseOp = qp.system.HamiltonianBase(Operators = Ops)
HBaseOp.write_info()

print "<H> = %s" % (HBaseOp.expectationValue(mf))
print "<T> = %s" % (T.expectationValue(mf))

I = qp.system.operators.Identity()
print "<I> = %s" % (I.expectationValue(mf))

# HBaseOp.apply(mf)

Fmf = qp.base.math.rs_to_fs(mf)
# print "norm(mf)=%s"%((mf.conjugate()*mf).sum()*mf.mesh.spacing)
# print "norm(Fmf)=%s"%((Fmf.conjugate()*Fmf).sum()*Fmf.mesh.FSspacing)
tmp =Fmf.mesh.FSpoints**2.0 * Fmf
# tmp =Fmf
# print Fmf
# print "tmp = %s" %(tmp.__class__.__name__)
# print tmp
Lmf = qp.base.math.fs_to_rs(tmp, return_mf = True)
# print "norm(Lmf)=%s"%((Lmf.conjugate()*Lmf).sum()*Lmf.mesh.spacing)
E = 0.5*(mf.conjugate()*Lmf).integrate()
# print "Class(E) = %s" %(E.__class__.__name__)
print "E = %s"%(E)

# ET = (mf.conjugate()*T.apply(mf)).integrate()
ET = T.expectationValue(mf)
print"ET = %s"%(ET)

if False:
    pl.rc('font', family='serif')
    pl.rc('font', size=12)
    pl.rc('legend', fontsize=10)

    fig = pl.figure()
    p1 = fig.add_subplot(1, 2, 1, title = 'Wavefunction - RS')
    p1.plot(mf.mesh.points, mf.real,  lw=1, color='r', label='$mf.real$')
    p1.plot(mf.mesh.points, mf.imag,  lw=1, color='b', label='$mf.imag$')
    p1.plot(mf.mesh.points, (mf.conjugate()*mf).real,  lw=2, color='g', label='$density$')
    # p1.plot(Lmf.mesh.points, (Lmf.conjugate()*Lmf).real,  lw=2, color='g', label='$density$')
    p1.legend()

    p2 = fig.add_subplot(1, 2, 2, title = 'Wavefunction - FS')
    p2.plot(Fmf.mesh.FSpoints, Fmf.real,  lw=1, color='r', label='$Fmf.real$')
    p2.plot(Fmf.mesh.FSpoints, Fmf.imag,  lw=1, color='b', label='$Fmf.imag$')
    p2.plot(Fmf.mesh.FSpoints, (Fmf.conjugate()*Fmf).real,  lw=2, color='g', label='$density$')
    # p2.plot(Fmf.mesh.FSpoints, Fmf.mesh.FSpoints,  lw=2, marker='o', color='g', label='$density$')
    p2.legend()

    pl.show()   
