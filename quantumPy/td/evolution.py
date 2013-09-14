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

from quantumPy.base      import messages
from quantumPy.grid      import mesh
from quantumPy.system    import hamiltonian
from quantumPy.system.hamiltonian  import Operator

from scipy.misc import factorial

printmsg = messages.print_msg  #shorter name         

class Propagator(Operator):
    """infinitesimal time propagator"""
    def __init__(self, Hamiltonian, **kwds):
        super(Propagator, self).__init__(**kwds)
        self.name = 'Propagator'
        
        self.Hamiltonian = Hamiltonian
        self.type = kwds.get('Type', 'exponential')
        
        
    def apply(self, psi, dt, t="0.0"):
        
        if   self.type.lower() == "exponential":
            exp(psi, self.Hamiltonian, t, dt)
        elif self.type.lower() == "aetrs":
            print "not implemented"
        else:
            raise "unknown option"    

    def write_info(self):    
        printmsg("# Time propagator: ")
        printmsg("#            type = %s"%(self.type))  


def exp(psiM, Hop, t, dt, order="5" ): 
    HpsiM = psiM
    UpsiM = psiM.copy()
    UpsiM[:] = 0.0
    for i in range(order):
        UpsiM +=  HpsiM * (- 1j * dt)**i /factorial(i)
        HpsiM = Hop.apply(HNpsi, t)      

    return UpsiM             