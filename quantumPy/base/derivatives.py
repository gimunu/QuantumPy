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

__all__=['fd_derivative', 'Derivative']

import numpy as np
from . import *
from ..grid.mesh import * 
# from ..base import math

# Finite differences coefficients weights. Naming convention:
# (derivative degree)_(number of points)p_(c center diffs / f forward diffs)

_fd_weight = {} 

#centered diffs
#1st derivative
_fd_weight['1_3p_c']=[-1./2., 0., 1./2.]
_fd_weight['1_5p_c']=[1./12., -2./3., 0., 2./3., -1./2.]
_fd_weight['1_7p_c']=[-1./60., 3./20., -3./4., 0., 3./4., -3./20., -1./60.]
_fd_weight['1_9p_c']=[-1./280., -4./105., 1./5., -4./5., 0., 4./5., -1./5., 4./105., -1./280.]
#2nd derivative
_fd_weight['2_3p_c']=[1., -2., 1.]
_fd_weight['2_5p_c']=[-1./12., 4./3., -5./2., 4./3., -1./12.]
_fd_weight['2_7p_c']=[1./90., -3./20., 3./2., 49./18., 3./2., -3./20., 1./90.]
_fd_weight['2_9p_c']=[-1./560., 8./315.,-1./5., 8./5., -205./72., 8./5., -1./5., 8./315., -1./560.]


class Derivative(object):
    """Numerical derivative class.
    
    This class handle the numerical derivative of MeshFunction(s).
    
    Attributes
    ----------
    mesh: Mesh
        The underlying mesh.
    bc: string 
        Boundary conditions (default = `zero').
        - `zero'    : zero boundary conditions.
        - `periodic': periodic bc.
    strategy: string
        The strategy used to perform the numerical derivation.
        - `fd': finite differences
        - `fs': Fourier-space
    order: integer
        The finite differences accuracy order (in grid spacing). Even orders up to 8 (default 2).
        Applies only when strategy = `fd'.
    order: Cube
        The cube mesh needed to perform FFT. 
        Applies only when strategy = `fs'.
    
    """
    def __init__(self, mesh, **kwds):
        super(Derivative, self).__init__()
        self.mesh = mesh
        
        self.bc =    kwds.get('Bc', 'zero')           # Boundary conditions
        self.strategy = kwds.get('Strategy', 'fs' if (self.bc.lower() == 'periodic') else 'fd')
        self.order = kwds.get('Order', 2)          
        self.cube = kwds.get('Cube', None)
        
        if (self.strategy.lower() == 'fs' and self.cube is None):
            self.cube = Cube(self.mesh, Attributes = 'RS + FS') 
    
    def perform(self, wfin, degree, dir = 1):
        if   (self.strategy.lower() == 'fd'):
            return self._fd_perform(wfin, degree, dir = dir)
        elif (self.strategy.lower() == 'fs'):
            return self._fs_perform(wfin, degree, dir = dir)
        else:
            raise Exception
            
            
    def _fs_perform(self, wfin, degree, dir = 1):
        # Use derivatives in Fourier space F(f'(x))=ik f(k), 
        if self.bc.lower() == 'zero':    
            raise NotImplementedError

        cf = mesh_to_cube(wfin, self.cube)
        Fcf = rs_to_fs(cf)
        K = Fcf.mesh.FSpoints        
        cf = fs_to_rs((- 1j * K)**degree * Fcf)
        wfout = cube_to_mesh(cf, wfin.mesh)
        
        return wfout
        
    def _fd_perform(self, wfin, degree, dir = 1):
        mesh = self.mesh
        
        # Handle exceptions
        if mesh.dim > 1:
            raise NotImplementedError
        
        if 'Cartesian' not in mesh.properties:    
            raise NotImplementedError

        if 'Uniform' not in mesh.properties:    
            raise NotImplementedError


        if   degree < 0:
            raise Exception('Unavailable derivative of order %s.'%(order))

        elif degree == 0:    
            return wfin

        elif degree == 1 or degree == 2:     
            wfout = wfin.copy()
            wfout[:] = 0.0
            
            if self.bc.lower() == 'periodic':    
                raise NotImplementedError
            
            n1 = mesh.np
            sp =  self.order + 1 # number of stencil points
            stname = '%d_%dp_c'%(degree, sp) # pick the appropriate weights   
            for ii in range(int(sp/2), n1 - int(sp/2)):
                for jj in range(sp):
                    wfout[ii] += wfin[ii - int(sp/2) + jj] * _fd_weight[stname][jj]
            wfout /= mesh.spacing**(degree)        
        else:
            raise Exception('Unavailable derivative of order >= 2. Order %s was given.'%(order))

        return wfout    
    
    def write_info(self, indent = 0):        
        print_msg ( "Derivatives:", indent = indent)
        print_msg( "       bc = %s "%(self.bc), indent = indent+1)
        print_msg( "strategy  = %s "%(self.strategy), indent = indent+1)

        if self.strategy.lower() == 'fd':
            print_msg( "  order = %d   (%d-points stencil)"%(self.order, self.order+1), indent = indent+2)   
        if self.strategy.lower() == 'fs':
            self.cube.write_info(indent = indent+2)



def fd_derivative(wfin, **kwds):
    """Finite difference derivative of a MeshFunction.
    
    ...
    
    """

    mesh = wfin.mesh
    boundary = kwds.get('Boundary', 'zero') # Boundary conditions
    degree = kwds.get('Degree', 1)          # Derivative degree (1 = 1s derivative, 2 =2nd derivative )
    order = kwds.get('Order', 2)            # Fine difference approx. order

    # Handle exceptions
    if mesh.dim > 1:
        raise NotImplementedError
        
    if 'Cartesian' not in mesh.properties:    
        raise NotImplementedError

    if 'Uniform' not in mesh.properties:    
        raise NotImplementedError


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
        stname = '%d_%dp_c'%(degree, sp) # pick the appropriate weights   
        for ii in range(int(sp/2), n1 - int(sp/2)):
            for jj in range(sp):
                wfout[ii] += wfin[ii - int(sp/2) + jj] * _fd_weight[stname][jj]
        wfout /= mesh.spacing**(degree)        
    else:
        raise Exception('Unavailable derivative of order %s.'%(order))

    return wfout    