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
        The hamiltonian operator. The choice is not restricted to the Hamiltonian class
        and other subclass of Operator can be employed.
    method : str
        The approximation method. Default value is 'exponential'.
    
    """
    def __init__(self, Hamiltonian, **kwds):
        super(Propagator, self).__init__(**kwds)
        self.name    = 'Time-propagator'
        self.symbol  = 'U(t, t+dt)'
        self.formula = '\exp(-i dt H(t))'
        
        self.H    = Hamiltonian
        self.method = kwds.get('Method', 'exponential')
        
        
    def applyRigth(self, wfinR, **kwds):

        dt   = kwds.get('dt', 0.01)
        time = kwds.get('time', 0.00)
        
        if   self.method.lower() == "exponential":
            exp(wfinR, self.H, time, dt)
            
        elif self.method.lower() == "aetrs":
            print "not implemented"

        else:
            raise Exception("unknown option")    

    def write_info(self, indent = 0): 
        from functools import partial
        printmsg = partial(messages.print_msg, indent = indent)       
        print_msg = messages.print_msg    
           
        printmsg( "%s (%s): "%(self.name, self.symbol) )       
        print_msg("%s = %s "%(self.symbol, self.formula), indent = indent+1)    
        print_msg("method = %s"%(self.method), indent = indent+1)  
        self.H.write_info(indent = indent+1)
        


def exp(psiM, Hop, time, dt, order = 5 ): 
    HpsiM = psiM
    UpsiM = psiM.copy()
    UpsiM[:] = 0.0
    for i in range(order):
        UpsiM +=  HpsiM * (- 1j * dt)**i /factorial(i)
        HpsiM = Hop.apply(HpsiM)
        # print i, HpsiM      

    return UpsiM             