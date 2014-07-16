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

"""QuantumPy output helper module."""

from __future__ import division

__all__ = ['output']

import warnings
import numpy as np


from .grid import *

def write(qpEntity, name = None, format ='gnuplot', **kwds):
    """Wrappers to write various quantumPy entities."""


    name = name if name else qpEntity.__class__.__name__

    if   isinstance(qpEntity, MeshFunction):
        write_meshFunction(qpEntity, name, **kwds)
        
    else:
        warnings.warn("no output for quantumPy entity: %s"% qpEntity.__class__.__name__)
        


def write_meshFunction(mf, filename, **kwds):
    """Write meshFunction."""
    
    
    file = None
    dim = mf.mesh.dim

    # format = kwds.get('format', "gnuplot")
    # if format.lower == "gnuplot":

    file = open("%s.gp"%filename,"w")

    if dim == 1:
        for i in range(mf.mesh.np):
            file.write("%02.8e\t%02.8e\t%02.8e\n"%(mf.mesh.points[i],mf[i].real,mf[i].imag))

    # else:
    #     warnings.warn("Not recognized write format %s for %s"% (format, mf.__class__.__name__))

        
    if file:    
        file.close()
    

    
    