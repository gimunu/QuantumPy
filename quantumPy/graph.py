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

try:
    import pylab as pl
except ImportError:
    pl = None

from .grid import *

def plot(qpEntity, **kwds):
    """Wrappers to plot various quantumPy entities."""

    out = None
    if isinstance(qpEntity, MeshFunction):
        out = plot_meshFunction(qpEntity, **kwds)
    else:
        raise Exception
        
    return out 

def plot_meshFunction(mf, **kwds):
    """docstring for plot_meshFunction"""
    
    dim = mf.mesh.dim
    out = None

    # No pylab no party
    if not pl:
        return out
        
    if dim == 1:
        
    else:
        raise Exception    
    
    return out
    