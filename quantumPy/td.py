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

__all__=['Propagator', 'propagator']

import numpy as np
from .base import *
from .grid.mesh import *
from .operators import *

from scipy.misc import factorial

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
        
        
    def apply(self, wfinR, side = 'R', **kwds):

        dt   = kwds.get('dt', 0.01)
        time = kwds.get('time', 0.0)
        
        if   self.method.lower() == "exponential":
            wfout = exp(wfinR, self.H, time, dt)
            
        elif self.method.lower() == "split":
            raise NotImplementedError

        else:
            raise Exception("unknown option")    

        return wfout    

    def write_info(self, indent = 0): 
        print_msg( "%s (%s): "%(self.name, self.symbol), indent = indent )       
        print_msg("%s = %s "%(self.symbol, self.formula), indent = indent+1)    
        print_msg("method = %s"%(self.method), indent = indent+1)  
        self.H.write_info(indent = indent+1)
        


def exp(wf, Hop, time, dt, order = 4 ): 
    """ Exponential operator """
    Hwf = wf
    Uwf = MeshFunction(np.zeros(wf.mesh.np, dtype = complex), wf.mesh)
    Uwf[:] = 0.0
    for i in range(order):
        Uwf +=  Hwf * (- 1j * dt)**i /factorial(i)
        Hwf = Hop.apply(Hwf)
        
    return Uwf             
    
def propagator(H, method = 'etrs', dt = 0.1, **kwds ):
    
    U = None
    if   method == 'exp':
        U = td_exp(H, order = kwds.get('exp_order', 4), dt = dt)
    elif method == 'etrs':    
        U = td_etrs(H, order = kwds.get('exp_order', 4), dt = dt)
    else:
        raise Exception("Unrecognized evolution method `%s'."%method)
        
    return U    

###########################################    
# Propagators
###########################################

def td_exp(H, order = 4, dt = 0.01):
        
    U = exponential(H, order = order, exp_step = dt)

    # wraps around exponential to set the step in the 
    # Taylor expansion dh = -i * dt
    uraction = U.raction
    def raction(wf, **kwds):
        kwds['exp_step']= -1j * kwds.get('dt', dt)  
        return uraction(wf, **kwds)

    ulaction = U.laction
    def laction(wf, **kwds):
        kwds['exp_step']= -1j * kwds.get('dt', dt)  
        return ulaction(wf, **kwds)

            
    U.set_action(laction, 'L')    
    U.set_action(raction, 'R')
    
    U.name    = 'Exponential propagation'
    U.symbol  = 'U_exp'
    U.formula = 'exp{-i dt %s(t)}'%(H.symbol)

    return U

#------------------------------------------

def td_etrs(H, order = 4, dt = 0.01):
    """Creates a time-reversal-symmetry (ETRS) based propagation Operator.
    
    ...
    
    
    Notes
    -----
    The ETRS operator is defined as:
    
    U_{ETRS}= \exp{\i \Delta t H(t+\Delta t)} \exp{-i \Delta t H(t)}     
    """
    
    ExpT   = td_exp(H, order = order, dt = dt)    
    expraction1 = ExpT.get_action('R') 
    def expTraction(wf, **kwds):
        #exp(-i dt/2 H(t - dt))
        kwds['dt']= kwds.get('dt', dt)/2.0  
        kwds['time']=  kwds.get('time', 0.0) - dt   
        return expraction1(wf, **kwds)
    ExpT.set_action(expTraction, 'R') 
            
    
    ExpTdT = td_exp(H, order = order, dt = dt)
    expraction2 = ExpTdT.get_action('R') 
    def expTdTraction(wf, **kwds):
        #exp(-i dt/2 H(t))
        _dt = kwds.get('dt', dt)  
        kwds['dt']=  _dt/2.0
        kwds['time']=  kwds.get('time', 0.0)   
        return expraction2(wf, **kwds)
    ExpTdT.set_action(expTdTraction, 'R') 
    
    U = ExpTdT * ExpT 
    
    U.name = "ETRS propagation"
    U.symbol  = 'U_etrs'
    U.formula = 'exp{-i dt/2 %s(t+dt)} * exp{-i dt/2 %s(t)}'%(H.symbol, H.symbol)
    U.info    = '%s = %s'%(H.symbol, H.formula)
    
    return U   
    
#------------------------------------------
# LASERS
#------------------------------------------

class ExternalField(object):
    """ExternalField class.
    
    Describes electro-magnetic external perturbations.
    """
    def __init__(self, sb, **kwds):
        super(ExternalField, self).__init__()
        
        self.type = kwds.get('type','electric_field')
        pol = [0.0]*sb.dim
        pol[0] = 1.0    
        self.pol      = kwds.get('polarization', pol)  
        self.omega    = kwds.get('omega', 1.0)
        self.envelope = kwds.get('envelope', tdf_constant(1.0))
        self.phase    = kwds.get('phase', tdf_constant(0.0))    
    
    def evaluate(self, time, **kwds):
        return  self.pol[:] * self.envelope(time) np.sin(self.omega *time + self.phase(time)) 
        
        

def tdf_constant(const):
    def tdf(time, **kwds):
        return const
    return tdf    
    
def tdf_trapezoidal(amplitude, tconst, tramp, tau):

    def tdf(time, **kwds):
        if   np.abs(time-tau) <= tconst:
            return amplitude
        elif np.abs(time-tau) <= tconst+tramp:
            return  (amplitude/tramp) * time + tconst  
        else:
            return 0.0    
                  
    return tdf        