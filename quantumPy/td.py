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
        self.sb = sb
        pol = [0.0]*sb.dim
        pol[0] = 1.0    
        self.pol      = np.array(kwds.get('polarization', pol)) 
        #normalize
        self.pol      = self.pol/np.sqrt(np.dot(self.pol, self.pol))
        self.omega    = kwds.get('omega', 1.0)
        self.envelope = kwds.get('envelope', tdf_constant(1.0))
        self.phase    = kwds.get('phase', tdf_constant(0.0))    
        # Grid values
        self.field    = None
        self.times    = None
            
    def _time_func(self, time, **kwds):
        return self.envelope.evaluate(time, **kwds) * \
                np.sin(self.omega * time + self.phase.evaluate(time, **kwds)) 
        
    def evaluate(self, time, **kwds):
        return  self.pol * self._time_func(time, **kwds)

    def grid_evaluate(self, times, **kwds):
        self.time  = np.array(times)
        self.field = np.zeros((self.time.size, self.pol.size)) 
        i=0
        for t in self.time:
            self.field[i,:] = self.evaluate(t, **kwds)
            i += 1
            
    def  write_info(self,  **kwds):
        indent = kwds.get('indent', 0)
        print_msg( "Laser: ", indent = indent ) 
        print_msg( "Polarization =  %s"%(self.pol), indent + 1)
        print_msg( "Carrier - w  = %1.4e [a.u]"%(self.omega), indent + 1)
        print_msg( "          l  = %1.4e [nm]"%(1240./(self.omega*27.211)), indent + 1)
        print_msg( " ", indent + 1)
        self.envelope.write_info(indent + 1)
        print_msg( " ", indent + 1)
        # optionally ask for a time-grid 
        times = kwds.pop('times', None)
        if times != None:
            self.grid_evaluate(times, **kwds)

        # 1 atomic unit of intensity = 6.4364086e+15 W / cm^2
        # In a Gaussian system of units,
        # I(t) = (1/(8\pi)) * c * E(t)^2
        # (1/(8\pi)) * c = 5.4525289841210 a.u.
        if self.field != None:
            max_intensity = 0
            max_field = 0
            fluence = 0
            for f in self.field:
                field = np.dot(f,f)
                intensity =  field * 5.4525289841210
                fluence = fluence + intensity    
                if(intensity > max_intensity): 
                    max_intensity = intensity 
                if(field > max_field):
                     max_field = field         
            fluence *= np.abs(self.time[1]-self.time[0])    
            Up = max_field/(4*self.omega**2)
            
            print_msg( "Peak intensity  = %1.4e [a.u]"%(max_intensity), indent + 1)
            print_msg( "                = %1.4e [W/cm^2]"%(max_intensity*6.4364086e+15), indent + 1)
            print_msg( "Int. intensity  = %1.4e [a.u]"%(fluence), indent + 1)
            print_msg( "Fluence         = %1.4e [a.u]"%(fluence/5.4525289841210), indent + 1)
            print_msg( "Ponderomotive E = %1.4e [a.u]"%(Up), indent + 1)
            
            
                
class TDfunction(object):
    """docstring for TDfunction"""
    def __init__(self, **kwds):
        super(TDfunction, self).__init__()
        self.func = kwds.get('function',None)
        self.name = kwds.get('name',None)
        self.info = kwds.get('info',None)
        
    def evaluate(self, time, **kwds):
        return self.func(time, **kwds)
    
    def write_info(self, indent = 0):
        print_msg( "%s: "%(self.name), indent = indent ) 
        print_msg( "%s"%(self.info), indent = indent+1 )                         


def tdf_constant(const):
    def func(time, **kwds):
        return const
    
    tdf = TDfunction()
    tdf.func = func     
    tdf.name = "Constant"     
    tdf.info = "amplitude = %1.5e "% const     
        
    return tdf    
    
def tdf_trapezoidal(amplitude, tconst, tramp, tau):

    def func(time, **kwds):
        f = 0
        if   time > tau - tconst/2. - tramp and time <= tau - tconst/2.:
            f = (time - (tau - tconst/2. - tramp))/tramp
        elif time > tau - tconst/2.  and time <= tau + tconst/2.:
            f = 1.0
        elif time > tau + tconst/2.  and time <= tau + tconst/2. + tramp:
            f =(tau + tconst/2. + tramp - time)/tramp 
        else:
            f = 0.0    
        return f*amplitude

    tdf = TDfunction()
    tdf.func = func     
    tdf.name = "Trapezoidal"     
    tdf.info = "amplitude  = %1.5e [a.u]\n\
tconst     = %1.5e [a.u]\n\
tramp      = %1.5e [a.u]"% (amplitude, tconst, tramp) 

                  
    return tdf        