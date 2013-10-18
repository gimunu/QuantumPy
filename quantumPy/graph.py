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

"""QuantumPy plotting helper module."""

from __future__ import division

__all__ = ['plot']

import warnings
import numpy as np

try:
    import pylab as pl
except ImportError:
    pl = None

from .grid import *

def plot(qpEntity, **kwds):
    """Wrappers to plot various quantumPy entities."""

    out = None
    if   isinstance(qpEntity, MeshFunction):
        out = plot_meshFunction(qpEntity, **kwds)
        
    elif isinstance(qpEntity, Mesh):
        out = plot_mesh(qpEntity, **kwds)

    else:
        warnings.warn("%s is not a quantumPy `plottable' entity"% qpEntity.__class__.__name__)
        
    return out 

def plot_meshFunction(mf, **kwds):
    """Plot meshFunction."""
    
    dim = mf.mesh.dim
    out = None

    # No pylab no party
    if not pl:
        return out
        
    if dim == 1:
        out = pl.plot(mf.mesh.points, mf, **kwds)
    else:
        raise Exception    
    
    return out

def plot_mesh(mesh, **kwds):
    """Plot Mesh points."""
    dim = mesh.dim
    out = None
    
    if not pl:
        return out


    if dim <= 2:
        data = np.zeros((mesh.i2c.shape[0], 2))
        data[:,0:dim]= mesh.i2c[:,0:dim]

        if kwds.pop('labels', False):
            labels = ['{0}'.format(i) for i in range(mesh.np)]
            for label, x, y in zip(labels, data[:,0], data[:,1]):
                pl.annotate(
                    label, 
                    xy = (x, y), xytext = (-2, 2),
                    textcoords = 'offset points', ha = 'right', va = 'bottom')
        
        out = pl.scatter(data[:,0], data[:,1], **kwds) 
        
                # pl.annotate(
                #     label, 
                #     xy = (x, y), xytext = (-20, 20),
                #     textcoords = 'offset points', ha = 'right', va = 'bottom',
                #     bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                #     arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

    else:
        raise Exception    
    
    return out
    
    