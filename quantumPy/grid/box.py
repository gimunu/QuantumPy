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

__all__=['Box', 'Cube', 'mesh_to_cube', 'cube_to_mesh']

import numpy as np
from mesh           import Mesh
from quantumPy.base import messages



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
            self.points = np.arange(- self.radius, self.radius + self.spacing, self.spacing)
            # self.points = np.linspace(- self.radius, self.radius, num = self.radius/self.spacing,  endpoint=True)
            
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
        printmsg = partial(messages.print_msg, indent = indent)       
         
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


def mesh_to_cube(mf, cube):
    """ Maps a mesh function into a cube function."""
    # if (mf.mesh.__class__.__name__ == 'Cube'):
    if ( isinstance(mf.mesh, Cube)):
        #do nothing
        return mf
        
    if (mf.mesh != cube.mesh):
        raise Exception ("Incompatible mesh and cube.")
        
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
    if (mesh != cube.mesh):
        raise Exception ("Incompatible mesh and cube.")
    
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
            
            kmax = np.pi/self.mesh.spacing
            dk = 2.0*np.pi/np.abs(xmax-xmin)

            self.FSspacing = dk 
            self.FSpoints  = np.arange(- kmax, kmax + dk, dk) 
            self.FSsize    = np.array([np.amin(self.FSpoints, 0),np.amax(self.FSpoints, 0)])
            
        else:
            raise Exception("Not implemented for dim>1")    

    def write_info(self, indent = 0):
        from functools import partial
        printmsg = partial(messages.print_msg, indent = indent)       

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