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
    """Infinitesimal time propagation operator.
    
    ...
    
    Attributes
    ----------
    H : Hamiltonian
        The hamiltonian operator.
    type : str
        The approximation method. Default 'exponential'
    
    """
    def __init__(self, Hamiltonian, **kwds):
        super(Propagator, self).__init__(**kwds)
        self.name    = 'Propagator'
        self.symbol  = 'U(t, t+dt)'
        self.formula = '\exp(-i dt H)'
        
        self.H    = Hamiltonian
        self.type = kwds.get('Type', 'exponential')
        
        
    def applyRigth(self, wfinR, **kwds):

        dt   = kwds.get('dt', 0.01)
        time = kwds.get('time', 0.00)
        
        if   self.type.lower() == "exponential":
            exp(wfinR, self.H, time, dt)
        elif self.type.lower() == "aetrs":
            print "not implemented"
        else:
            raise "unknown option"    

    def write_info(self, indent = 0): 
        from functools import partial
        printmsg = partial(messages.print_msg, indent = indent)       
        print_msg = messages.print_msg    
           
        printmsg("Time propagator:")
        printmsg(" " ) 
        print_msg("type = %s"%(self.type), indent = indent+1)  
        self.H.write_info(indent = indent+1)
        


def exp(psiM, Hop, time, dt, order = 5 ): 
    HpsiM = psiM
    UpsiM = psiM.copy()
    UpsiM[:] = 0.0
    for i in range(order):
        UpsiM +=  HpsiM * (- 1j * dt)**i /factorial(i)
        HpsiM = Hop.apply(HpsiM)
        print i, HpsiM      

    return UpsiM             