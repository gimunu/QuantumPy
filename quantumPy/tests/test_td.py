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

def imaginary_time_scf(U, H, wfin, dt = 0.01 , Nmax = 1e4, Eth = 1e-6):

    dtau = -1j * dt
    
    wft = wfin.copy()
    Eold = 0.0
    for i in range(0, int(Nmax)):
        # Apply the propagator
        wft = U.apply(wft, dt = dtau)         
        norm2 =  (wft*wft.conjugate()).integrate().real
        wft /= np.sqrt(norm2)
        # Calculate the energy
        E   = H.expectationValue(wft)
        dE  = np.abs(Eold - E)/np.abs(E)
        Eold = E
        if (dE < Eth):
            print "Converged!"
            break

    return E

def test_propagators():
    Radius = 4.0
    box = qp.box(shape = 'Sphere', radius = Radius, spacing = 0.1)
    
    def vext(x):
        omega = 1.0
        return 0.5*(x*omega)**2
    
    H = qp.hamiltonian(box, vext)    

    wfin = qp.grid.MeshFunction( np.random.rand(box.np)+0.j, mesh = box)

    U = qp.td.propagator(H, method ='etrs')
    E_etrs = imaginary_time_scf(U, H, wfin, dt = 0.01)  
    assert_almost_equal(E_etrs, 0.500,  decimal=4 )
    

    U = qp.td.propagator(H, method ='exp')
    E_etrs = imaginary_time_scf(U, H, wfin, dt = 0.005)    
    assert_almost_equal(E_etrs, 0.500,  decimal=3)
    
    
    
    