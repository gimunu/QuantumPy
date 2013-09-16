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

__all__=['Mesh', 'MeshFunction']

import numpy as np
from quantumPy.base import messages
from quantumPy.base import types


class Mesh(object):
    """Basic Mesh class"""
    def __init__(self, **kwds):
        super(Mesh, self).__init__()
        
        self.dim        = kwds.get('Dim', 1)
        self.points     = kwds.get('Points', np.array([]))
        self.properties = kwds.get('Properties', 'Uniform + Cartesian')
        if (self.__class__.__name__ == 'Mesh'):  #Avoid name-clash in subclasses 
            self.update()  
        
    def update(self):
        """Refresh all the data-dependent values after a change"""
        self.np = self.points.size 
    
    def write_info(self, indent):
        from functools import partial

        printmsg = partial(messages.print_msg, indent = indent)       
        printmsg = messages.print_msg        
        printmsg("Mesh info: " )
        printmsg("       properties = %s "%(self.properties))        
        printmsg("       dimensions = %d "%(self.dim))        
        printmsg("           points = %d "%(self.np))        



class MeshFunction(np.ndarray):
    """Subclass of numpy.ndarray specifying a function defined on the mesh.
       Each instance of the class appends a mesh attribute to an ndarray and must be initialized this way:
       
       ar = np.zeros(mesh.np)
       mf = MeshFunction(ar, mesh = mesh)
       
       A instance of the class can also be crated with copy():
       
       mf_copy = mf.copy()
       
    """
    
    def __new__(cls, input_array, mesh=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.mesh = mesh
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self,obj):
        # reset the attribute from passed original object
        self.mesh = getattr(obj, 'mesh', None)
        # We do not need to return anything

    def integrate(self, Region = None):
        """ Performs the integration of the function on the underlying mesh.
            The integration is by default on the full mesh where the function is defined,
            an optional sub-mesh domain can be entered with the Region option. 
        """
        if ('Uniform' in self.mesh.properties and 'Cartesian' in self.mesh.properties):
            out = self.sum(dtype=types.CMPLX) * self.mesh.spacing            
        
        return out.astype(types.CMPLX)

