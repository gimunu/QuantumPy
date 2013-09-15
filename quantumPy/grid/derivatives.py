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
from quantumPy.base import messages
from quantumPy.grid.mesh import Mesh, MeshFunction



# Finite differences coefficients weights. Naming convention:
# (derivative degree)_(dimension)d_(number of points)p_(c center diffs / f forward diffs)

_fd_weight = {} 

#centered diffs
#1st derivative
_fd_weight['1_1d_3p_c']=[-1./2., 0., 1./2.]
_fd_weight['1_1d_5p_c']=[1./12., -2./3., 0., 2./3., -1./2.]
_fd_weight['1_1d_7p_c']=[-1./60., 3./20., -3./4., 0., 3./4., -3./20., -1./60.]
_fd_weight['1_1d_9p_c']=[-1./280., -4./105., 1./5., -4./5., 0., 4./5., -1./5., 4./105., -1./280.]
#2nd derivative
_fd_weight['2_1d_3p_c']=[1., -2., 1.]
_fd_weight['2_1d_5p_c']=[-1./12., 4./3., -5./2., 4./3., -1./12.]
_fd_weight['2_1d_7p_c']=[1./90., -3./20., 3./2., 49./18., 3./2., -3./20., 1./90.]
_fd_weight['2_1d_7p_c']=[-1./560., 8./315.,-1./5., 8./5., -205./72., 8./5., -1./5., 8./315., -1./560.]


class FD_Derivatives(object):
    """Finite differences derivatives class.
    """
    def __init__(self, mesh, **kwds):
        super(FD_Derivatives, self).__init__()
        self.mesh = mesh
        


def fd_derivative(wfin, **kwds):
    """Finite difference derivative for MeshFunction."""

    mesh = wfin.mesh
    boundary = kwds.get('Boundary', 'zero') # Boundary conditions
    degree = kwds.get('Degree', 1)          # Derivative degree (1 = 1s derivative, 2 =2nd derivative )
    order = kwds.get('Order', 2)            # Fine difference approx. order

    # Handle exceptions
    if mesh.dim > 1:
        raise Exception('Unavailable derivative in dimension > 1.')
        
    if 'Cartesian' not in mesh.properties:    
        raise Exception('Non-cartesian meshes are not supported.')

    if 'Uniform' not in mesh.properties:    
        raise Exception('Non-uniform meshes are not supported.')


    if   degree < 0:
        raise Exception('Unavailable derivative of order %s.'%(order))

    elif degree == 0:    
        return wfin

    elif degree == 1 or degree == 2:     
        wfout = wfin.copy()
        wfout[:] = 0.0
        #1D
        n1 = mesh.np
        sp =  order + 1 # number of stencil points
        stname = '%d_%dd_%dp_c'%(degree, mesh.dim, sp)    
        for ii in range(sp/2, n1 - sp/2):
            for jj in range(sp):
                wfout[ii] += wfin[ii - sp/2 + jj] * _fd_weight[stname][jj]
        wfout[:] /= mesh.spacing**(degree)        
    else:
        raise Exception('Unavailable derivative of order %s.'%(order))

    return wfout    