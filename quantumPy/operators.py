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
import types
import numpy as np
from scipy.misc import factorial
from .base import *
from .base import trees
from .grid import *



# Flatten nested lists
def flatten(*args):
    for x in args:
        if hasattr(x, '__iter__'):
            for y in flatten(*x):
                yield y
        else:
            yield x

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
        self.expr = BinaryTree(self)
        
        self.hermitian = True
        self.raction = None
        self.laction = None
        
        self.update()
    
    def action(self, funct, key = 'LR'):
        """Define Operator action.
         
        """
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
        self.hermitian = (self.raction == self.laction)
        
        # Generate a formula representation        
        self.formula = str(self.expr)


    def has_left_action(self):
        return self.laction != None
                    
    def has_right_action(self):
        return self.raction != None

    
    #
    # Action of MeshFucntion(s)
    #    
    
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

        out = (wfL * self.apply(wfR, side = 'R', **kwds)).integrate()

        return out 
            
    def apply(self, wfin, side= 'R', **kwds):
        """ Apply the operator to a wavefunction.
        
        By default this is the right-action `O * Wf'.
        
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

        _apply = self._applyLR(side)
        
        return _apply(self, wfin, **kwds)

    
                        
    def _evaluate(self, expr, wf, side, **kwds):
        """Evaluates the algebraic expression containing operators. 
        
        This is the core method that runs through the expression tree and 
        performs the operations on the correct order.
        
        """
        # This is the main method that evaluates the tree expression.
        # It is recursive with a loop spanning different methods.
        # This is the loop:
        # 1. apply()
        # 2. _applyLR()
        # 3. _evaluate
        # 4. _action_[add, sub, mul]()
        # in point 4 there is explicit reference to apply() (point 1)
        # and the loop closes.
        
        
        opers = {'+':'_action_add', '-':'_action_sub', '*':'_action_mul'}
        
        debug_msg("---- _evaluate (%s) "%side)
        
        def get_operator_from(node):
            if   node == None:
                raise Exception("Don't know what to do!")
            elif isinstance(node.getRootVal(), Operator):
                Op = node.getRootVal()
            else:
                Op = Operator()
                Op.expr = node
            return Op    
        
        nodeL = None
        nodeR = None
        if expr:
            # In order to perform the right (left) action we start by applying 
            # the action of the operator form the right-most (left-most) leaf of the tree 
            # on the right (left) together with the sibling node.
            
            if   side == 'R':
                nodeR = expr.getRightChild()
                nodeL = nodeR.getSibling()
            elif side == 'L':                    
                nodeL = expr.getLeftChild()
                nodeR = nodeL.getSibling()
            else: 
                raise Exception("Which side is `%s'?"%side)
            
            debug_msg("nodeL = %s"%nodeL)
            debug_msg("nodeR = %s"%nodeR)
            
            OpL = get_operator_from(nodeL)
            OpR = get_operator_from(nodeR)            

            operation = getattr(self, opers[expr.getRootVal()])
            debug_msg("return = %s %s %s"%(OpL.expr, expr.getRootVal(),OpR.expr ))
            debug_msg("----")
            return  operation(OpL, OpR, wf, side, **kwds)
                        

                    
    
    def _applyLR(self, side):
        """Left and Right application generation.
        
        """ 
        debug_msg("called _applyLR (side = %s) from %s"%(side, self.expr))
        
        if   side =='L':
            actionattr  = 'laction'
        elif side == 'R':    
            actionattr  = 'raction'
        else:
            raise Exception('Unrecognized action %s'%side)
        
        def actionLR(self, wfin, **kwds):

            wfout = MeshFunction(np.zeros(wfin.mesh.np, dtype = wfin.dtype), wfin.mesh)

            if getattr(self, actionattr):
                action = getattr(self, actionattr)
                wfout = action(wfin, **kwds)
            else:
                wfout = self._evaluate(self.expr, wfin, side, **kwds)
        
            return wfout

        
        return actionLR
        
    
 
    def _action_add(self, OpL, OpR, wfin, side, **kwds):
         wfout  = OpL.apply(wfin, side, **kwds) + OpR.apply(wfin, side, **kwds)
         return wfout

    def _action_sub(self, OpL, OpR, wfin, side, **kwds):
         wfout  = OpL.apply(wfin, side, **kwds) - OpR.apply(wfin, side, **kwds)
         return wfout

    def _action_mul(self, OpL, OpR, wfin, side, **kwds):
        
         # wfout = MeshFunction(np.zeros(wfin.mesh.np, dtype = wfin.dtype), wfin.mesh)
         
         if side == 'R':
             wfout = OpR.apply(wfin , side, **kwds)
             wfout = OpL.apply(wfout, side, **kwds)
         else:
             wfout = OpL.apply(wfin , side, **kwds)
             wfout = OpR.apply(wfout, side, **kwds)            
         
         return wfout


    #
    # Operator info
    #
        
    def __str__(self):
        return self.symbol

    def _write_inorder(self, tree, indent = 0):
        if tree.leftChild:
            self._write_inorder(tree.leftChild, indent = indent)
        if isinstance(tree.key, Operator):
            tree.key._write_details(indent)
        if tree.rightChild:
            self._write_inorder(tree.rightChild, indent = indent)
    
    def _write_details(self, indent = 0): 
        print_msg( "%s operator (%s): "%(self.name, self.symbol), indent = indent )       
        if self.formula != None:
            print_msg( "%s = %s "%(self.symbol, self.formula), indent = indent+1)    
        if self.info != None:
            print_msg( "info: %s "%(self.info), indent = indent+1)    
            
    def write_info(self, indent = 0):
        self._write_details(indent = indent)
        # Write details of all the composing ops
        # for Op in self.op_list:
        #     Op.write_info(indent = indent+1)
        
        self._write_inorder(self.expr, indent = indent+1)

    #
    # Override operators: +, -, * .
    #

    def _is_scalar_element(self, x):
    		"""Is x a scalar?

    		"""
    		return isinstance(x, types.IntType) or \
    				isinstance(x, types.FloatType) or \
    				isinstance(x, types.ComplexType)    
                    
    def _algebraic_operation( self, other, operation, reverse = False):

        opers = ['+','-','*']
        assert(operation in opers)
        
        if self._is_scalar_element(other):
            other = scalar(other)
        
        elif not isinstance(other, Operator):
            raise TypeError("Cannot perform '%s' and type %s" %(operation, type(other)))
            
        Op = Operator()
        Op.name    = 'Composed-operator'
        Op.symbol  = 'O'
        
        Op.expr.setRootVal(operation)
        if reverse:
            Op.expr.insertRight(self.expr)
            Op.expr.insertLeft(other.expr)
        else:    
            Op.expr.insertLeft(self.expr)
            Op.expr.insertRight(other.expr)

        Op.update()        
        return Op
        

    def __add__(self, other):
        return    self._algebraic_operation(other, '+', reverse = False)
   
    def __radd__(self, other):
        return    self._algebraic_operation(other, '+', reverse = True)
        
    
    def __mul__(self, other):        
        return    self._algebraic_operation(other, '*', reverse = False)
        
    def __rmul__(self, other):
        return    self._algebraic_operation(other, '*', reverse = True)
        
    def __neg__(self):
        return -1*self 
           
    def __sub__(self, other):    
        return    self._algebraic_operation(other, '-', reverse = False)
        
    def __rsub__(self, other):
        return    self._algebraic_operation(other, '-', reverse = True)
            
        


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

#------------------------------------------------------
def scalar(val):
    """Identity operator."""

    def action(wf, **kwds):
        return val*wf

    Op = Operator()
    Op.name    = 'Scalar'
    Op.symbol  = str(val)
    Op.formula = str(val)
        
    Op.action(action, 'LR')    
    return Op


#------------------------------------------------------
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


#------------------------------------------------------
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
        def action(wf, **kwds):
            dh = kwds.get('exp_step', 1.0)
            Opinwf = wf
            Uwf = MeshFunction(np.zeros(wf.mesh.np, dtype = complex), wf.mesh)
            Uwf[:] = 0.0
            for i in range(order):
                Uwf +=  Opinwf * (dh)**i / factorial(i)
                Opinwf = Opin.apply(Opinwf, side, **kwds)        
            return Uwf

        return action

        
    Op = Operator()
            
    Op.action(get_action('L'), 'L')    
    Op.action(get_action('R'), 'R') 

    Op.name    = 'Exponential'
    Op.symbol  = 'Exp'
    Op.formula = 'exp(%s)'%Opin.symbol
    Op.info    = '%s = %s'%(Opin.symbol, Opin.formula)
    
    #Needed for write_info()
    Op.expr = Opin.expr
       
    return Op
    


#------------------------------------------------------
class Gradient(Operator):
    """Gradient operator"""
    def __init__(self, mesh, **kwds):
        super(Gradient, self).__init__(**kwds)
        self.name    = 'Gradient'
        self.symbol  = '\\nabla'
        self.formula = 'd/dx'
        
        self.der = kwds.get('Der', Derivative(mesh, **kwds)) 
    
    def apply(self,wfin, side='R', **kwds): 
        return self.der.perform(wfin, degree = 1)        
        
    def write_info(self, indent = 0):
        super(Gradient,self).write_info(indent = indent)
        self.der.write_info(indent = indent)
        


#------------------------------------------------------
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
    
    def apply(self,wfin, side='R', **kwds):     
        if side == 'R':
            return self.der.perform(wfin, degree = 2)
        else: 
            return self.apply(wfin.conjugate(), side='R', **kwds)              
                
    def write_info(self, indent = 0):        
        super(Laplacian,self).write_info(indent)
        self.der.write_info(indent = indent)


#------------------------------------------------------
class Kinetic(Laplacian):
    """Kinetic operator"""
    def __init__(self, mesh, **kwds):
        super(Kinetic,self).__init__(mesh, **kwds)
        self.name    = 'Kinetic'
        self.symbol  = 'T'
        self.formula = '1/2 \\nabla^2'
    
    
    def apply(self, wfin, side = 'R', **kwds):
        return -0.5*super(Kinetic,self).apply(wfin, side = side, **kwds)    
    


#------------------------------------------------------
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
        