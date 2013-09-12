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

import numpy as np
from quantumPy.base import messages
from quantumPy.base.math import rs_to_fs, fs_to_rs
from quantumPy.grid.mesh import Mesh, MeshFunction
from quantumPy.grid.box  import Cube, Box
from quantumPy.grid.box  import mesh_to_cube, cube_to_mesh
from quantumPy.system.operators import Operator, Laplacian, Kinetic

printmsg = messages.print_msg  #shorter name         

#############################################
#
#############################################
class HamiltonianBase(Operator):
    """Basic Hamiltonian operator"""
    def __init__(self, **kwds):
        super(HamiltonianBase, self).__init__(**kwds)
        self.name   = "Hamiltonian"
        self.symbol = "H"
        self.time = kwds.get('Time', 0.0)
        self.operators = kwds.get('Operators', [])

    def applyRigth(self,wfin):    
        wfout    = wfin.copy()
        wfout[:] = 0.0 
        for Op in self.operators:
            wfout += Op.applyRigth(wfin)
        return wfout

    def applyLeft(self,wfin):    
        wfout    = wfin.copy()
        wfout[:] = 0.0
        for Op in self.operators:
            wfout += Op.applyLeft(wfin)
        return wfout
        
    def write_info(self, indent = 0):
        from functools import partial
        printmsg = partial(messages.print_msg, indent = indent)       
        print_msg = messages.print_msg    
        
        printmsg( "Hamiltonian info: " )
        printmsg( " " ) 
        Hstring = '%s = '%self.symbol 
        for Op in self.operators:
            Hstring = Hstring + Op.symbol + ' + ' 
        print_msg( "%s"%(Hstring), indent = indent + 1)        
        printmsg( " " ) 
        for Op in self.operators:
            Op.write_info()


#############################################
#
#############################################
class Hamiltonian(HamiltonianBase):
    """Hamiltonian class with simplified interface"""
    def __init__(self, **kwds):
        super(Hamiltonian, self).__init__(**kwds)


