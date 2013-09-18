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

__all__ = ['Operator', 'Laplacian', 'Gradient', 'Kinetic', 'identity',
           'scalar_pot' , 'Hamiltonian', 'exponential']

import inspect
import numpy as np
from scipy.misc import factorial
from .base import *
from .grid import *


#############################################
#
#############################################
class Operator(object):
    """General low-level operator acting on a MeshFunction.
    
    This one of the main classes of the package. 
    
    Attributes
    ----------
    name : str
        Operator long literal name.
    symbol : str
        Operator symbol (LaTeX format).
    formula : str
        Operator formula (LaTeX format).
    hermitian : bool
        Whether the operator is Hermitian or not.
    [R,L]action : callable
        The right (left) action.
    op_list : list of Operator instances
        A list of operators composing the operator.
    
    """
    def __init__(self, **kwds):
        super(Operator, self).__init__()        
        self.name    = kwds.get('name'   , "Operator")
        self.symbol  = kwds.get('symbol' , "Op")
        self.formula = kwds.get('formula', self.symbol)
        self.info    = kwds.get('info'   , None)
        
        self.op_list = kwds.get('Operators', [])
        
        self.hermitian = True
        self.raction = None
        self.laction = None
        
        self.update()
    
    def action(self, funct, key = 'LR'):
        
        if (key =='LR'):
            self.raction = funct
            self.laction = funct
        elif (key == 'L'):
            self.laction = funct    
        elif key == 'R':
            self.raction = funct 
        else:
            raise Exception

        self.update()    

             
    def update(self):
        """Update the internal attributes.
        
        This method should be called in order to regenerate all the internal attribute
        dependencies.
        
        """
        
        # Generate a formula representation
        if len(self.op_list) > 0:
            self.formula = ''
        
        for i in range(len(self.op_list)):
            Op = self.op_list[i]
            self.formula = self.formula + Op.symbol  
            if i < len(self.op_list)-1:
                self.formula = self.formula + ' + '
            
    def apply(self, wfin, **kwds):
        """ Apply the operator to a wavefunction.
        
        ...
        
        Parameters
        ----------
        wfin : MeshFunction
            Function acting from the right.
        **kwds : dictionary
            Optional additional parameters. This allow a subclass to extend the method.
            
        Returns
        -------    
        wfout: MeshFunction
            The function resulting from the operator application.
        
        """
        return  self.applyRight(wfin, **kwds)

    def expectationValue(self, wfR, wfL = None, **kwds):
        """ Calculate the expectation value of the operator.
        
        ...
        
        Parameters
        ----------
        wfR : MeshFunction
            Function acting from the right.
        wfL : MeshFunction
            Function acting from the left.
        **kwds : dictionary
            Optional parameters for subclass extension.
        
        Returns
        -------    
        out: complex 
            The operator expectation value.
        
        """
        
        if wfL == None:
            wfL = wfR.copy().conjugate()

        out = (wfL * self.applyRight(wfR, **kwds)).integrate()

        return out 
        
        
    def applyRight(self, wfin, **kwds): 
        """Operator right action.
        
        Compose the right action by sequentially applying all the operators in the 
        operators list.
        
        Parameters
        ----------
        wfin : MeshFunction
            Function acting from the right.
        **kwds : dictionary
            Optional parameters for subclass extension.

        Returns
        -------    
        wfout: MeshFunction
            The function resulting from the right operator application.
        
        """   

        wfout = MeshFunction(np.zeros(wfin.mesh.np, dtype = wfin.dtype), wfin.mesh)

        if self.raction != None:
            wfout = self.raction(wfin, **kwds)
        else:
            for Op in self.op_list:
                wfout += Op.applyRight(wfin, **kwds)
        
        return wfout

    def applyLeft(self, wfin, **kwds):    
        """Operator left action.
        
        Compose the right action by sequentially applying all the operators in the 
        operators list.
        
        Parameters
        ----------
        wfin : MeshFunction
            Function acting from the left.
        **kwds : dictionary
            Optional parameters for subclass extension.
        
        Returns
        -------    
        wfout: MeshFunction
            The function resulting from the left operator application.
        
        """   

        wfout = MeshFunction(np.zeros(wfin.mesh.np, dtype = wfin.dtype), wfin.mesh)

        if self.laction != None:
            wfout = self.laction(wfin, **kwds)
        else:
            for Op in self.op_list:
                wfout += Op.applyLeft(wfin, **kwds)
        
        return wfout


    def write_info(self, indent = 0):
        print_msg( "%s operator (%s): "%(self.name, self.symbol), indent = indent )       
        if self.formula != None:
            print_msg( "%s = %s "%(self.symbol, self.formula), indent = indent+1)    
        if self.info != None:
            print_msg( "info: %s "%(self.info), indent = indent+1)    
        # Write details of all the composing ops
        for Op in self.op_list:
            Op.write_info(indent = indent+1)


#############################################
#  Library of operators
#############################################

def identity():
    """Identity operator."""

    def action(wf, **kwds):
        return wf

    Op = Operator()
    Op.name    = 'Identity'
    Op.symbol  = 'I'
    Op.formula = '1'
        
    Op.action(action, 'LR')    
    return Op


def scalar_pot(func):
    """Scalar potential operator."""

    def action(wf, **kwds):
        r = wf.mesh.points
        return func(r) * wf

    Op = Operator()
    Op.name    = 'Scalar potential'
    Op.symbol  = func.__name__
    Op.formula = ''
    Op.info = '\n'+inspect.getsource(func)
        
    Op.action(action, 'LR')    
    return Op


def exponential(Opin, order = 4, exp_step = 1.0):
    """Exponential operator.
           
    Creates the exponential operator of operator Opin 
    exp(Opin) with a Taylor expansion at a given order.

    Parameters
    ----------
    Opin: quantumPy.Operator
        The operator to exponentiate.
    order: int
        The order of the Taylor expansion.
    exp_step: float or complex
        The step size of the expansion.
        
    Returns
    -------
    Op: quantumPy.Operator
        The exponential operator exp(Opin). 
        
    Notes
    -----
    The Taylor expansion is made with an additional step parameter `dh'
    as follow:
    
    \exp(O dh) = sum_{i=0}{order} (O dh)^i/i!
    
    This parameter can also be specified when operator is applied by specifying 
    `exp_step' parameter:

    Examples:
    ---------
    >>> ...
    >>> Exp = qp.exponential(Op)
    >>> ewf = Exp.apply(wf, exp_step = 0.1)
    >>> ...        
    """
    
    def get_action(side):
        if   side == 'L':
            OpinApply = Opin.applyLeft
        elif side == 'R':    
            OpinApply = Opin.applyRight 
        else: 
            raise Exception

        def action(wf, **kwds):
            dh = kwds.get('exp_step', 1.0)
            Opinwf = wf
            Uwf = MeshFunction(np.zeros(wf.mesh.np, dtype = complex), wf.mesh)
            Uwf[:] = 0.0
            for i in range(order):
                Uwf +=  Opinwf * (dh)**i / factorial(i)
                Opinwf = OpinApply(Opinwf, **kwds)        
            return Uwf

        return action

        
    Op = Operator()
    Op.name    = 'Exponential'
    Op.symbol  = 'Exp'
    Op.formula = 'exp(%s)'%Opin.symbol
    
    #Needed for write_info()
    Op.op_list = [Opin]
        
    Op.action(get_action('L'), 'L')    
    Op.action(get_action('R'), 'R') 
       
    return Op
    


#############################################
class Gradient(Operator):
    """Gradient operator"""
    def __init__(self, mesh, **kwds):
        super(Gradient, self).__init__(**kwds)
        self.name    = 'Gradient'
        self.symbol  = '\\nabla'
        self.formula = 'd/dx'
        
        self.der = kwds.get('Der', Derivative(mesh, **kwds)) 
        
    def applyRight(self, wfinR, **kwds): 
        return self.der.perform(wfinR, degree = 1)


    def write_info(self, indent = 0):
        super(Gradient,self).write_info(indent = indent)
        self.der.write_info(indent = indent)
        

#############################################
class Laplacian(Operator):
    """Laplacian operator"""
    def __init__(self, mesh, **kwds):
        super(Laplacian, self).__init__(**kwds)
        
        self.name    = 'Laplacian'
        self.symbol  = '\\nabla^2'
        self.formula = 'd^2/dx^2'
        if mesh.dim > 1:
            self.formula += '+ d^2/dy^2' 
        if mesh.dim > 2:
            self.formula += '+ d^2/dz^2' 
        
        self.der = kwds.get('Der', Derivative(mesh, **kwds)) 
        
                
    def applyRight(self,wfinR, **kwds):        
        return self.der.perform(wfinR, degree = 2)
         
    def applyLeft(self,wfinL, **kwds):
        return self.applyRight(wfinL.conjugate())      

    def write_info(self, indent = 0):        
        super(Laplacian,self).write_info(indent)
        self.der.write_info(indent = indent)

#############################################
class Kinetic(Laplacian):
    """Kinetic operator"""
    def __init__(self, mesh, **kwds):
        super(Kinetic,self).__init__(mesh, **kwds)
        self.name    = 'Kinetic'
        self.symbol  = 'T'
        self.formula = '1/2 \\nabla^2'
        
    def applyRight(self, wfinR, **kwds): 
        return -0.5*super(Kinetic,self).applyRight(wfinR)
    

#############################################
class Hamiltonian(Operator):
    """Hamiltonian operator.
    
    Utility class to create an Hamiltonian.
    
    Attributes
    ----------
    time : float
        Current time. Used when a time-dependent term is included.
    
    """
    def __init__(self, **kwds):
        super(Hamiltonian, self).__init__(**kwds)
        self.name   = "Hamiltonian"
        self.symbol = "H"
        self.time = kwds.get('Time', 0.0)
        