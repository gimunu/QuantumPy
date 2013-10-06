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

__all__=['Derivative']

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

_stencils = [[0.],

            [[0.],
            [0., 1./2.],
            [0., 2./3., -1./2.],
            [0., 3./4., -3./20., -1./60.],
            [0., 4./5., -1./5., 4./105., -1./280.]],

            [[0.],
            [-2., 1.],
            [-5./2., 4./3., -1./12.],
            [-49./18., 3./2., -3./20., 1./90.],
            [-205./72., 8./5., -1./5., 8./315., -1./560.],
            [-5269./1800., 5./3., -5./21., 5./126., -5./1008., 1./3150.],
            [-5369./1800., 12./7., -15./56., 10./189., -1./112., 2./1925., -1./16632.]]]

            # [-5269./1800., 5./3., -5./21., 5./126., -5./1008., 1./3150.],
            # [-5369./1800., 12./7., -15./56., 10./189., -1./112., 2./1925., -1./16632.]],

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
        self.order = kwds.get('Order', 4)          
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
        cf = fs_to_rs((1j * K)**degree * Fcf)
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
            return wfin.copy()

        # elif degree == 1  or degree == 2: 
        #     wfout = wfin.copy()
        #     wfout[:] = 0.0
        #     
        #     # print "weird implementation"
        #     
        #     if self.bc.lower() == 'periodic':    
        #         raise NotImplementedError
        #     n1 = mesh.np
        #     sp =  2*self.order + 1 # number of stencil points
        #     stname = '%d_%dp_c'%(degree, sp) # pick the appropriate weights   
        #     for ii in range(int(sp/2), n1 - int(sp/2)):
        #         for jj in range(sp):
        #             wfout[ii] += wfin[ii - int(sp/2) + jj] * _fd_weight[stname][jj]
        #     wfout /= mesh.spacing**(degree)        

        elif degree == 1 or degree == 2: 
            # print "hopefully better implementation"
            
            
            n1 = mesh.np
            dh = mesh.spacing
            sp =  int(self.order + 1) # number of stencil points
            sp1 = sp - 1
            st_deg = _stencils[degree]
            stencil = st_deg[self.order]
            if   degree == 1:
                st2 =  np.concatenate(([-x for x in stencil[::-1]],stencil[1:]))
            elif degree == 2:
                st2 =  np.concatenate((stencil[::-1],stencil[1:]))
            
            #inner region
            # print stencil
            
            wfout = MeshFunction(np.convolve(wfin, st2[::-1], mode='same'), mesh = wfin.mesh)
            
            # wfout = wfin.copy()
            # wfout[:] = 0.0
            #     
            # for ii in range(sp1, n1 - sp1):
            #     wfout[ii] += wfin[ii] * stencil[0]
            #     for jj in range(1, sp):
            #         wfout[ii] += (wfin[ii + jj] + wfin[ii - jj]) * stencil[jj]
            
            # # boundaries
            # if self.bc.lower() == 'zero':
            #     for ii in range(sp - 1):
            #         # II = n1 - ii -1
            #         wfout[ii] += wfin[ii] * stencil[0]
            #         # wfout[II] += wfin[II] * stencil[0]
            #         for jj in range(1, sp): 
            #             wfout[ii] += wfin[ii + jj] * stencil[jj]
            #             # wfout[II] += wfin[II - jj] * stencil[jj]
            #         for jj in range(1, sp-ii): 
            #             wfout[ii] += wfin[ii - jj] * stencil[jj]
            #             # wfout[II] += wfin[II + jj] * stencil[jj]
            # #     
            # # elif self.bc.lower() == 'periodic':    
            # #     raise NotImplementedError
            
                            
            wfout /= dh**(degree)        
            
            
            # # Use Ask's    
            # wfout = second_derivative(wfin, self.order, periodic= (self.bc == 'periodic'))
            # wfout = MeshFunction(wfout, mesh = wfin.mesh )

        else:
            raise Exception('Unavailable derivative of order >= 2. Order %s was given.'%(order))

        assert (isinstance(wfout, wfin.__class__))
        
        return wfout    
    
    def write_info(self, indent = 0):        
        print_msg ( "Derivatives:", indent = indent)
        print_msg( "       bc = %s "%(self.bc), indent = indent+1)
        print_msg( "strategy  = %s "%(self.strategy), indent = indent+1)

        if self.strategy.lower() == 'fd':
            print_msg( "  order = %d   (%d-points stencil)"%(self.order, 2*self.order+1), indent = indent+2)   
        if self.strategy.lower() == 'fs':
            self.cube.write_info(indent = indent+2)






# def get_kinetic(x, order=1, periodic=False):
def second_derivative(wf, order=1, periodic=False):
    """ Ask's implementation"""
    stencils = [[0.],
                [-2., 1.],
                [-5./2., 4./3., -1./12.],
                [-49./18., 3./2., -3./20., 1./90.],
                [-205./72., 8./5., -1./5., 8./315., -1./560.],
                [-5269./1800., 5./3., -5./21., 5./126., -5./1008., 1./3150.],
                [-5369./1800., 12./7., -15./56., 10./189., -1./112., 2./1925.,
                 -1./16632.]]

    N = wf.mesh.np
    dx = wf.mesh.spacing
    T = np.zeros((N, N), dtype=complex)
    Tflat = T.ravel()

    stencil = stencils[order]

    for i, val in enumerate(stencil):
        Tflat[i::N + 1] = val
        Tflat[i * N::N + 1] = val
    if periodic:
        if order > 1:
            raise NotImplementedError
        T[-1, 0] = 1.0
        T[0, -1] = 1.0
    else:
        for tmp in [T[:, -1], T[:, 0], T[0, :], T[-1, :]]:
            tmp[:] = 0.0
    T *= 1.0 / dx**2
    return np.dot(T,wf)
