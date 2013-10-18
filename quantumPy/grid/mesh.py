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

__all__=['Mesh', 'MeshFunction','box', 'Cube', 'mesh_to_cube', 'cube_to_mesh', 'submesh', 'SubMesh', 
         'sphere', 'segment']

import numpy as np
import functools
from ..base.messages import *
from ..base import types


#############################################
#
#############################################
class Mesh(object):
    """Mesh class.
    
    The spatial grid defining the real space sampling points.
    
    Attributes
    ----------
    dim: integer
        The space dimensionality.
    points: array of tuples
        The set of point coordinates defining the mesh.
    np: integer
        The global number of points.    
    properties: str    
        String defining the low level geometrical properties of the mesh.
        The options are: Uniform, Cartesian.
    info: dictionary
        Dictionary containing additional information specific of an instance
        such as the geometry.
        
    """
    
    def __init__(self, **kwds):
        super(Mesh, self).__init__()
        
        self.dim        = kwds.get('dim', 1)
        self.spacing    = kwds.get('spacing', 0.1) 
        
        # index to coordinate map
        self.i2c        = np.array([])
        # coordinate to index map
        self.c2i        = np.array([])
        self.points     = self.i2c
        
        self.properties = kwds.get('properties', 'Uniform + Cartesian')
        self.info       = None
        if (self.__class__.__name__ == 'Mesh'):  #Avoid name-clash in subclasses 
            self.update()  
        
    def update(self):
        """Refresh all the data-dependent values after a change"""
        # self.np = self.i2c.shape[0] #if self.dim > 1 else self.i2c.ravel().size 
        self.np = self.points.size if self.dim == 1 else  self.i2c.shape[0]
    
    def write_info(self, indent = 0):
        from functools import partial

        printmsg = partial(print_msg, indent = indent)       
        # printmsg = messages.print_msg        
        printmsg("Mesh info: " )
        printmsg("       properties = %s "%(self.properties))        
        printmsg("       dimensions = %d "%(self.dim))        
        printmsg("           points = %d "%(self.np))
        if self.info:
            print_msg(self.info, indent = indent+1)
        

    def get_coords_from_index(self, idx):
        return self.i2c[idx,0:self.dim]

    def __str__(self):
        return "<mesh: np = %d>"%(self.np)

    def __contains__(self, item):
        for p in self.i2c:
                if p is item:
                    return True
        return False

    def __iter__(self):
        for p in self.i2c:
                yield p


class SubMesh(Mesh):
    """SubMesh class.
    
    Defines a subset of a parent mesh.
    
    Attributes
    ----------
    mesh: Mesh obj
        Reference to the parent mesh object.
    pindex: integer array    
        Index set containing the indices of mesh points defining the submesh. 
        
    """
    def __init__(self, mesh):
        super(SubMesh, self).__init__()
        self.mesh = mesh
        self.pindex = np.array([], dtype=int)



def submesh(func, mesh):
    smesh = SubMesh(mesh)
    for ii in range(mesh.np):
        if func(mesh.points[ii]):
            smesh.pindex = np.append(smesh.pindex, int(ii))
    return smesh

#############################################
# Geometrical shapes
#############################################

def sphere(pos, Radius=1.0, dim = 1, center = [0,0,0]):
    p = np.array(pos) 
    c = np.array(center) 
    r2 = np.sum(np.power(p[0:dim]-c[0:dim],2))

    return r2 <= Radius**2        


def disk(pos, Radius=1.0):
    return sphere(pos, Radius = Radius, dim = 2)

    
def segment(pos, P1, P2, dim = 1):
    if dim == 1:
        x = pos
        return x >= P1 and x <= P2     


def cube(pos, L, dim = 1, center = [0,0,0]):
    if   dim == 1:
        LL = (L)
    elif dim == 2:
        LL = (L,L)
    elif dim == 3:             
        LL = (L,L,L)
    return cuboid(pos, LL, dim = dim, center = center)

    
def cuboid(pos, L, dim = 1, center = (0,0,0)):
     if   dim == 1:
          x,  = pos
          Lx, = L 
          cx, = center
          return x - cx <= Lx/2. and x - cx >= -Lx/2.
     elif dim == 2:
          x, y = pos
          Lx, Ly = L 
          cx, cy = center
          boundx = x - cx <= Lx/2. and x - cx >= -Lx/2.
          boundy = y - cy <= Ly/2. and y - cy >= -Ly/2.
          return boundx and boundy
    
     elif dim == 3:     
          x, y, z = pos
          Lx, Ly, Lz = L 
          cx, cy, cz = center
          boundx = x - cx <= Lx/2. and x - cx >= -Lx/2.
          boundy = y - cy <= Ly/2. and y - cy >= -Ly/2.
          boundz = z - cz <= Lz/2. and z - cz >= -Lz/2.
          return boundx and boundy and boundz
    
     else:
         return False     


#############################################
def box(shape, coord = 'cartesian',  **kwds):
    """Create a simulation box."""
    
    box = Mesh(**kwds)

    box.info =            "Geometry: \n"
    
    if callable(shape):
        box.info = box.info + "      Shape = Custom\n" 
        fshapewargs = shape
        
    else:       
        box.info = box.info + "      Shape = %s\n"%shape 
        if   shape.lower() == 'sphere': 
            fshapewargs = functools.partial(sphere,  Radius = kwds.get('radius'), dim = box.dim)
            box.info = box.info + "     Radius = %1.2e\n"%kwds.get('radius')    
        elif shape.lower() == 'cube':
            fshapewargs = functools.partial(cube,    L = kwds.get('Lsize'), dim = box.dim)
            box.info = box.info + "      Lsize = %1.2e\n"%kwds.get('Lsize')    
        elif shape.lower() == 'cuboid':
            fshapewargs = functools.partial(cuboid,  L = kwds.get('Lsize'), dim = box.dim)
        else: 
            raise Exception
    
    box.info = box.info + "    spacing = %1.2e\n"%box.spacing    
        
        
    p = [0.0]*box.dim
    points = floodFill(p, fshapewargs, box.spacing, coordinate(coord), dim = box.dim)

    box.points = np.sort(points.ravel()) if box.dim == 1 else points
    box.i2c    = np.sort(points) if box.dim == 1 else points
    
    
    box.update()
    
    return box



def floodFill_stack(p, func, step, coord, pts = None, dim = 1):
    """ Flood fill.

    Recursive stack-based implementation.
    """
    first = False
    if pts == None:
        pts = np.array([p])
        first = True
        
    
    pinpts = any((pts[:]== p).all(1))
    
    if (not func(p) or pinpts) and not first:
        # print "func(p = %e) = %e"%(p, func(p))
        # debug_msg("p %s is already in pts %s or hitting edge - return"%(p,  pts), lev = DEBUG_VERBOSE)
        return pts
    else:
        # debug_msg("p %s"%p, lev = 10)
        pinpts = any((pts[:]== p).all(1))
        if  not pinpts:
            pts = np.append(pts, [p], axis = 0)
            # debug_msg("p %s added to pts %s"%(p, pts), lev = DEBUG_VERBOSE)         

        for idir in range(1, dim+1):
            # move forward along dir
            # debug_msg("move fwd dir = %d"%idir, lev = DEBUG_VERBOSE)
            pts = floodFill(coord.next(p, step,  idir, dim), func, step, coord, pts = pts, dim = dim)
            # move backward along dir
            # debug_msg("move bwd dir = %d"%idir, lev = DEBUG_VERBOSE)
            pts = floodFill(coord.next(p, step, -idir, dim), func, step, coord, pts = pts, dim = dim)

    return pts

def floodFill(p, inshape, step, coord,  dim = 1):
    """ Iterative flood fill.
    """
    toFill = list()
    toFill.append(p)
    pts = None

    while len(toFill) > 0:
        p = toFill.pop()
        
        if not inshape(p):
            continue
            
        pinpts = any((pts[:]== p).all(1)) if pts != None else False

        if not pinpts:
            pts = np.append(pts, [p], axis = 0) if pts != None  else np.array([p])
                        
            for idir in range(1, dim+1):
                # move forward along dir
                toFill.append(coord.next(p, step,  idir, dim))
                # move backward along dir
                toFill.append(coord.next(p, step, -idir, dim))

    return pts



class CoordinateGenerator(object):
    """Infinitesimal CoordinateGenerator class."""
    def __init__(self, name):
        super(CoordinateGenerator, self).__init__()
        self.name  = name
        self.nextf = None
        
    def next(self, pt, step, dir, dim, **kwds):
        """Generates next point.
        
        Starting from `pt', generates the next point along the direction `dir' at `step' distance.
        The sign of `dir' indicates wether to move forward `+' or backward `-'.        
        """
        if not callable(self.nextf):
            raise Exception("Coordinate next method not defined.")
            
        return self.nextf(pt, step, dir, dim, **kwds)
        

def coordinate(type):
    
    coord = None
    
    if type == 'cartesian':
        
        def next(pt, step, dir, dim):
            sgn = np.sign(dir)                
            # if   dim == 1:
            #     pt, = pt
            #     nxtpt = (round(pt/step) + sgn) * step, 
            # elif dim == 2: 
            #     if abs(dir) == 1:    
            #         nxtpt = [(round(pt[0]/step) + sgn) * step , pt[1]]
            #     else:    
            #         nxtpt = [pt[0], (round(pt[1]/step) + sgn) * step] 
            # elif dim == 3: 
            idir = abs(dir)-1 
            nxtpt = pt[:]
            nxtpt[idir] =  (round(pt[idir]/step) + sgn) * step
            
            # debug_msg("%s nxtpt %s "%(type, nxtpt), lev = DEBUG_VERBOSE)
            return nxtpt#[0:dim]
            
        coord = CoordinateGenerator(type)
        coord.nextf = next
        
    else:
        raise NotImplemented("Unrecognized coordinate type `%s'."%type)    
    
    return coord

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
            if Region:
                out = self[Region.pindex].sum() 
            else:    
                out = self.sum()
            
            out *= self.mesh.spacing               
        
        return out.astype(self.dtype)

    def norm2(self, Region=None):
        return (self.conjugate()*self).integrate(Region = Region)            

    def norm(self, Region=None):
        return np.sqrt(self.norm2(Region = Region))

