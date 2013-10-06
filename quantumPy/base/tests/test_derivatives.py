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

def gaussian_wp(mesh, sigma, k):
    X = mesh.points
    wf = qp.grid.MeshFunction( np.pi**(-1./4.) * sigma**(-1./2.) *  
                               np.exp(-X[:]**2. / (2.*sigma**2.) + 1j*k*X[:]) , mesh = mesh)
    return wf   

def D1gaussian_wp(mesh, sigma, k):
    X = mesh.points
    wf = gaussian_wp(mesh, sigma, k)
    wf[:] *= 1j*k - X [:]/sigma**2.
    return wf   

def D2gaussian_wp(mesh, sigma, k):
    X = mesh.points
    wf = gaussian_wp(mesh, sigma, k)
    wf[:] *= -(- X [:]**2. + sigma**2. + 2.*1j*k*X[:]*sigma**2. + k**2.*sigma**4.)/sigma**4.
    return wf   



def test_fd_derivatives():
    Radius = 4.0
    box = qp.Box(shape = 'Sphere', radius = Radius, spacing = 0.01)
    
    k = 4.0
    sigma = np.sqrt(2.0)/abs(k)
    wf = gaussian_wp(box, sigma, k)


    # D=qp.base.Derivative(box, Strategy = 'fs', Bc = 'periodic')
    D=qp.base.Derivative(box, Strategy = 'fd', Order = 4)
    D.write_info()

        
    D1wf = D.perform(wf, degree =1)
    D2wf = D.perform(wf, degree =2)

    D1wfex = D1gaussian_wp(box, sigma, k)
    D2wfex = D2gaussian_wp(box, sigma, k)


    ED1ex= (wf.conjugate()*D1wfex).integrate()
    ED1= (wf.conjugate()*D1wf).integrate()

    assert_almost_equal(ED1, ED1ex,  decimal=11)

    ED2ex= (wf.conjugate()*D2wfex).integrate()
    ED2= (wf.conjugate()*D2wf).integrate()

    assert_almost_equal(ED2, ED2ex,  decimal=11)
    
    
    
    