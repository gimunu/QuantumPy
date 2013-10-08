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

__all__=['Mesh', 'MeshFunction','Box', 'Cube', 'mesh_to_cube', 'cube_to_mesh']

import numpy as np
from ..base.messages import *
from ..base import types


#############################################
#
#############################################
class Mesh(object):
    """General Mesh class.
    
    ...
    
    
    """
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
    
    def write_info(self, indent = 0):
        from functools import partial

        printmsg = partial(print_msg, indent = indent)       
        # printmsg = messages.print_msg        
        printmsg("Mesh info: " )
        printmsg("       properties = %s "%(self.properties))        
        printmsg("       dimensions = %d "%(self.dim))        
        printmsg("           points = %d "%(self.np))        


#############################################
#
#############################################
class MeshFunction(np.ndarray):
    """Subclass of numpy.ndarray specifying a function defined on the mesh.
    
    ...
    
    Parameters
    ----------
    mesh : Mesh 
       A instance of Mesh.

    Examples
    --------
    Each instance of the class appends a mesh attribute to an ndarray and must be initialized this way:
    >>> a = np.zeros(mesh.np)
    >>> mf = MeshFunction(a, mesh = mesh)
    A instance of the class can also be crated with copy():
    >>> mf_cpy = mf.copy()       
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


#############################################
#
#############################################

class Box(Mesh):
    """The simulation box geometries.
       For the moment it's trivial we are working only in 1D.
    """
    def __init__(self, shape = 'Sphere', **kwds):
        super(Box,self).__init__(**kwds)

        self.shape   = shape
        self.spacing = kwds.get('spacing', 0.1) 

        if   self.shape.lower() == "sphere":
            self.radius = kwds.get('radius', 0.0)
                        
        elif self.shape.lower() == "rect":
            self.side = kwds.get('side', 0.0)
            
        else:
            raise "unknown option"
        
        if (self.__class__.__name__ == 'Box'):  #Avoid subclass name clash
            self.update()          

    def update(self):
        """Update derived data when some property changes"""

        if   self.shape.lower() == "sphere":
            # 1D
            
            def sphere(pos, R=self.radius, dim = 1):
                if   dim == 1:
                    x = pos
                    return x**2 < R**2
                elif dim == 2:
                    x, y = pos     
                    return x**2 + y**2 < R**2
                elif dim == 3:                
                    x, y, z = pos     
                    return x**2 + y**2 + z**2 < R**2
                return 
            
            points = floodFill(0.0, sphere, self.spacing)
            
            self.points = np.sort(points) if self.dim == 1 else points
            # self.points = np.arange(- self.radius, self.radius + self.spacing, self.spacing)
            
            
        elif self.shape.lower() == "rect":
            # 1D
            self.points = np.arange(- self.side/2.0, self.side/2.0 + self.spacing, self.spacing) 
            # self.points = np.linspace(- self.side/2.0, self.side/2.0 , num = self.side/self.spacing,  endpoint=True) 
                       
        else:
            raise Exception("unknown option")        
                
        super(Box, self).update()

        # Mesh.update(self) # refresh super        
        
    def write_info(self, indent = 0):
        from functools import partial
        printmsg = partial(print_msg, indent = indent)       
         
        printmsg( "Box info: " )       
        printmsg( "            shape = %s "%(self.shape))
        printmsg( "         spacing  = %1.2e [a.u.]"%(self.spacing))
        if   self.shape.lower() == "sphere":
            printmsg( "          radius  = %1.2e [a.u.]"%(self.radius))
        elif self.shape.lower() == "rect":
            printmsg( "            side  = %1.2e [a.u.]"%(self.side))
        else:
            raise Exception("unknown option")
            
        super(Box, self).write_info(indent)

def floodFill(p, func, step, pts = None, dim = 1):
    """ Flood fill.
        
    Stack-based recursive implementation.
    """
    if pts == None:
        pts = np.array([])
    
    if not func(p) or p in pts:
        return pts
    else:
        if dim == 1:        
            pts = np.append(pts, p)    
            pts = floodFill(p + step, func, step, pts = pts, dim = dim)
            pts = floodFill(p - step, func, step, pts = pts, dim = dim)
        else:
            raise NotImplemented

    return pts


#############################################
#
#############################################

def mesh_to_cube(mf, cube):
    """ Maps a mesh function into a cube function."""
    if ( isinstance(mf.mesh, Cube)):
        #do nothing
        return mf
    
    #FIXME: 
    # This test started to fail after the changes performed int the
    # Operator class to support expression evaluation.
    # The suspect is that this is due to some deepcopy effect  
    # and the fact the comparison '!=' is too stringent (e.g. compares pointers).    
    # if (mf.mesh != cube.mesh):
    #     raise Exception ("Incompatible mesh and cube.")
        
    if mf.mesh.dim == 1:
        out = mf.copy()
        out.mesh = cube
        
    else:
        raise Exception("Not implemented for dim>1")

    return out 

def cube_to_mesh(cf, mesh):
    """ Maps a cube function into a mesh function."""
    if ( isinstance(mesh, Cube)):
        #do nothing
        return cf
    
    cube = cf.mesh
    # if (mesh != cube.mesh):
    #     raise Exception ("Incompatible mesh and cube.")
    
    if mesh.dim == 1:
        out = cf.copy()
        out.mesh = mesh

    else:
        raise Exception("Not implemented for dim>1")

    return out 


class Cube(Mesh):
    """A cubic mesh. 
    
       This is a special mesh used for convenience operations such as write 3D wfs on a file 
       and perform FFTs.
       It is bound to a specific Mesh instance from which back and forth maps are defined. 
    """

    aliases = {
        'points'   : 'RSpoints',
        'spacing'  : 'RSspacing',
        }

    def __init__(self, mesh, **kwds):
        super(Cube, self).__init__(**kwds)
        self.attributes = kwds.get('Attributes', 'RS + FS')
        self.mesh = mesh  # The reference mesh
        self.dim  = kwds.get('Dim', mesh.dim)
        self.properties  = 'Uniform + Cartesian'
        
        if (self.__class__.__name__ == 'Cube'):  #Avoid subclass name clash
            self.update()          

    def update(self):
        self.__update_RS()
        if ('FS' in self.attributes):
            self.__update_FS()
        super(Cube, self).update()    
                    
    def __update_RS(self):
        if self.mesh.dim == 1:
            self.RSpoints  = self.mesh.points
            self.RSspacing = self.mesh.spacing
            self.RSsize    = np.array([np.amin(self.RSpoints, 0),np.amax(self.RSpoints, 0)])
        else:
            raise Exception("Not implemented for dim>1")    

    def __update_FS(self):
        if self.mesh.dim == 1:
            xmax = np.amax(self.mesh.points, 0)
            xmin = np.amin(self.mesh.points, 0)
            
            dk = 2.0*np.pi/np.abs(xmax-xmin)

            self.FSspacing = dk 
            nn = self.mesh.np
            self.FSpoints = np.fft.fftfreq(nn, d =1/(dk*nn)) 
            self.FSpoints = np.fft.fftshift(self.FSpoints)
             
            self.FSsize    = np.array([np.amin(self.FSpoints, 0),np.amax(self.FSpoints, 0)])
            
        else:
            raise Exception("Not implemented for dim>1")    

    def write_info(self, indent = 0):
        from functools import partial
        printmsg = partial(print_msg, indent = indent)       

        printmsg("Cube info: " )
        printmsg("       attributes = %s "%(self.attributes)) 
        if ('RS' in self.attributes):
            printmsg("  Real-Space ")                         
            printmsg("       dimensions = (%1.2e, %1.2e) [a.u.]"%(self.RSsize[0],self.RSsize[1]))        
            printmsg("          spacing = %1.2e [a.u.]"%(self.RSspacing))        
        if ('FS' in self.attributes):
            printmsg("  Fourier-Space ")                         
            printmsg("       dimensions = (%1.2e, %1.2e) [a.u.]"%(self.FSsize[0],self.FSsize[1]))        
            printmsg("          spacing = %1.2e [a.u.]"%(self.FSspacing))        

        # super(Cube, self).write_info(indent)

    # In order to implement the aliases on class attributes
    def __setattr__(self, name, value):
        name = self.aliases.get(name, name)
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name == "aliases":
            raise AttributeError  # http://nedbatchelder.com/blog/201010/surprising_getattr_recursion.html
        name = self.aliases.get(name, name)
        #return getattr(self, name) #Causes infinite recursion on non-existent attribute
        return object.__getattribute__(self, name)
