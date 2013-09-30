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
from numpy.testing import assert_equal, assert_almost_equal
import quantumPy as qp

def test_expression():
    Radius = 4.0
    box = qp.Box(shape = 'Sphere', radius = Radius, spacing = 0.1)
    
    wf = qp.grid.MeshFunction( np.random.rand(box.np)+0.j, mesh = box)
    wf /= np.sqrt((wf*wf.conjugate()).integrate().real)
    
    def vext(x):
        return 1.0 + 3.*x**2 -2.*x**3                               
    
    Vext = qp.scalar_pot(vext)
    T = qp.Kinetic(box)

    I = qp.identity() 
    
    O =  2*T + Vext*2 - 2*(T + Vext) + I -4.0 + 3.0 +1 

    Owf = O.expectationValue(wf)
    # print "<wf|O|wf>  = %s"%Owf  
    assert_almost_equal(Owf, 1.0)  


def test_kinetic():
    box = qp.Box(shape = 'Sphere', radius = 5.0, spacing = 0.1)
    
    X = box.points
    k = 4.0
    sigma = np.sqrt(2.0)/k
    wf = qp.grid.MeshFunction( np.pi**(-1.0/4.0) * sigma**(-1.0/2.0) *  
                               np.exp(-X[:]**2.0 / (2.0*sigma**2.0) + 1j*k*X[:]) , mesh = box)
    
    Tfs = qp.Kinetic(box, Strategy = 'fs', Bc = 'periodic')
    # print "<wf|Tfs|wf>  = %s"%Tfs.expectationValue(wf)   
    assert_almost_equal(Tfs.expectationValue(wf), 10.201) 

    Tfd = qp.Kinetic(box, Strategy = 'fd', Order = 5, Bc = 'zero')
    # print "<wf|Tfd|wf>  = %s"%Tfd.expectationValue(wf)
    assert_almost_equal(Tfd.expectationValue(wf), 9.99997705) 
    
    
