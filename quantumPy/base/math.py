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

__all__=['rs_to_fs', 'fs_to_rs']

from ..grid import * 

import numpy as np
from scipy.fftpack import fft, ifft, fftshift, ifftshift


#FFTs wrappers    
#Note: this code is 1D only (for now)
def rs_to_fs(wfin, cube_in = None):
    """ Transform a real-space mesh-function into Fourier-space.
    
    Parameters
    ----------
    wfin: MeshFunction
        The wavefunction to transfom.
    cube_in: Cube (default None)
        Optional Cube mesh to be used to perform FFTs.
        
    Returns
    -------
    wfout: MeshFunction
        A function defined on the Cube mapping to the underlying Mesh of wfin.
    
    Notes
    -----
    The interface is quite flexible as is accepts both mesh-functions and cube-functions.
    The output function is always a cube-function (i.e. mesh-function defined on the 
    special mesh Cube).
    Since the FFTs are always performed on cube-functions  a mapping is involved.
    An optional cubic mesh can be specified by setting cube_in.
    """

    #check whether the possible cube-mesh candidates are indeed Cube instances
    # and whether the cube Fourier-space (FS) has been initialized 
    cube = cube_in if (isinstance(cube_in, Cube) and ("FS" in getattr(cube_in, "attributes", ""))) else None
    wfin_is_cf = True if (isinstance(wfin.mesh, Cube) and ("FS" in getattr(wfin.mesh, "attributes", ""))) else False

    if (wfin_is_cf and cube == None):
        #Use wfin cube mesh
        cube = wfin.mesh 
        
    if cube == None:
        # Create a new cube
        # This is possiply resource consuming if performed several times and 
        # should be avoided if possible
        cube = Cube(wfin.mesh)
    
    cfin = mesh_to_cube(wfin, cube) if(not wfin_is_cf) else wfin    
    out = fftshift(fft(wfin))
    
    if wfin.mesh.dim == 1:
        # out /= np.sqrt(wfin.size*np.pi)
        out /= np.sqrt(wfin.mesh.np*np.pi/2)
    else:
        #Will see but for the moment
        raise Exception("Not implemented")

    out = MeshFunction(out, mesh = cube)
    
        
    return out

def fs_to_rs(wfin, return_mf = False, mesh_in = None):
    """ Transform from Fourier-space into Real-space.
        This function acts on cube-functions and returns cube functions.
    """
    assert(isinstance(wfin.mesh, Cube))

    cube = wfin.mesh
    
    out = ifft(ifftshift(wfin))

    if cube.dim == 1:
        # out = out * np.sqrt(wfin.size*(np.pi))
        out *= np.sqrt(wfin.mesh.np*np.pi/2)

    else:
        raise Exception("Not implemented")    
    
    out = MeshFunction(out, mesh = cube)
        
    if return_mf:
        mesh = mesh_in if (mesh_in != None) else cube.mesh    
        out = cube_to_mesh(out, mesh)
        
    return out 