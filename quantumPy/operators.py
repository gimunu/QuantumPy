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

__all__ = ['Operator', 'Laplacian', 'Gradient', 'Kinetic', 'kinetic', 'identity', 'scalar',
           'scalar_pot' , 'hamiltonian', 'exponential']

import inspect
import types
import copy
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
    
    def set_action(self, funct, key = 'LR'):
        """Define Operator action.
         
        """
        if (key =='LR' or key == 'RL'):
            self.raction = funct
            self.laction = funct
        elif (key == 'L'):
            self.laction = funct    
        elif key == 'R':
            self.raction = funct 
        else:
            raise Exception

        self.update()    
        
    def get_action(self, key):    
        if   key == 'R':
            return self.raction
        elif key == 'L':    
            return self.laction
        else:
            raise Exception    
             
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
        
 
                        
    def _evaluate(self, expr, wf, side, **kwds):
        """Evaluates the algebraic expression containing operators. 
        
        This is the core method that runs through the expression tree and 
        performs the operations on the correct order.
        
        """
        # This is the main method that evaluates the tree expression.
        # It is recursive with a loop cycle spanning different methods.
        # This is the loop:
        # 1. apply()
        # 2. _applyLR()
        # 3. _evaluate
        # 4. _action_[add, sub, mul]()
        # in point 4 a call to to apply() (cfr. 1.) closes the loop.
        
        
        opers = {'+':'_action_add', '-':'_action_sub', '*':'_action_mul'}
        
        debug_msg("---- _evaluate = %s [side %s] "%(expr, side))
        
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
            
            debug_msg("nodeL = %s - [parent = %s]"%(nodeL, nodeL.parent))
            debug_msg("nodeR = %s - [parent = %s]"%(nodeR, nodeR.parent))
            
            OpL = get_operator_from(nodeL)
            OpR = get_operator_from(nodeR)            

            operation = getattr(self, opers[expr.getRootVal()])
            debug_msg("perform: %s %s %s"%(OpL.expr, expr.getRootVal(),OpR.expr ))
            debug_msg("----")
            return  operation(OpL, OpR, wf, side, **kwds)
                    
    
    def _action_add(self, OpL, OpR, wfin, side, **kwds):
         # Note: the order of application of the operators is important
         # as it defines the starting point in the recursive evaluation 
         # of the expression tree. 
         # For instance if we want to evaluate from right we always need to 
         # apply to the right operator before the left one.

         debug_msg("in  _action_add: |wf|^2=%s"%(wfin.conjugate()*wfin).integrate())        
         if side == 'R':
             wfout  = OpR.apply(wfin, side, **kwds)
             debug_msg ("outR _action_add: 1 < %s > = %s "%(OpR.expr, (wfin.conjugate()*wfout).integrate()))        
             wfout += OpL.apply(wfin, side, **kwds)
             debug_msg("outL _action_add: 2 < %s + %s > = %s "%(OpL.expr, OpR.expr, (wfin.conjugate()*wfout).integrate()))        
         else:
             wfout  = OpL.apply(wfin, side, **kwds)
             wfout += OpR.apply(wfin, side, **kwds)
            
         return wfout


    def _action_sub(self, OpL, OpR, wfin, side, **kwds):
         debug_msg ("in  _action_sub: |wfin|^2=%s "%(wfin.conjugate()*wfin).integrate())        
         if side == 'R':
             wfout  = - OpR.apply(wfin, side, **kwds)
             debug_msg("outR _action_sub: 1 < -%s > = %s "%(OpR.expr, (wfin.conjugate()*wfout).integrate()))        
             wfout += OpL.apply(wfin, side, **kwds)
             debug_msg("outL _action_sub: 2 < %s - %s > = %s "%(OpL.expr,OpR.expr, (wfin.conjugate()*wfout).integrate()))
         else:
             wfout  = OpL.apply(wfin, side, **kwds)
             wfout -= OpL.apply(wfin, side, **kwds)
         return wfout


    def _action_mul(self, OpL, OpR, wfin, side, **kwds):
         debug_msg ("_action_mul: |wf|^2=%s"%(wfin.conjugate()*wfin).integrate())        
         
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
        if self.formula:
            print_msg( "%s = %s "%(self.symbol, self.formula), indent = indent+1)    
        if self.info:
            print_msg( "info: %s "%(self.info), indent = indent+1)    
            
    def write_info(self, indent = 0):
        self._write_details(indent = indent)
        
        # Write details of all the composing ops        
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
        Op.name    = 'Composed'
        Op.symbol  = 'O'
        
        # If the expression contain two (or more) times an operator
        # this may cause loops or broken branches in the tree caused by 
        # Op.expr being the same for different nodes. 
        # In order to avoid it we perform a deepcopy of each Operator in the 
        # expression. 
        Op.expr.setRootVal(operation)
        cself  = copy.deepcopy(self)
        cother = copy.deepcopy(other)
        
        if reverse:
            Op.expr.insertRight(cself.expr)
            Op.expr.insertLeft( cother.expr)
        else:    
            Op.expr.insertLeft(  cself.expr)
            Op.expr.insertRight(cother.expr)

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
    
    def __div__(self, other):
        if self._is_scalar_element(other):
            return    self._algebraic_operation(1.0/other, '*', reverse = True)
        
    def __neg__(self):
        return -1*self 
           
    def __sub__(self, other):    
        return    self._algebraic_operation(other, '-', reverse = False)
        
    def __rsub__(self, other):
        return    self._algebraic_operation(other, '-', reverse = True)
            
        


#############################################
#  Library of Operators
#  --------------------
#
#  How do I create new operators?
#
#  There are basically two way one can follow in order to create new operators:
#  1. Subclassing Operator class.
#  2. Define a function returning an Operator instance.
#
#############################################


def identity():
    """Identity operator."""

    def action(wf, **kwds):
        return wf.copy()

    Op = Operator()
    Op.name    = 'Identity'
    Op.symbol  = 'I'
    Op.formula = '1'
        
    Op.set_action(action, 'LR')    
    return Op

#------------------------------------------------------
def scalar(val):
    """Identity operator."""

    def action(wf, **kwds):
        return val * wf.copy()

    Op = Operator()
    Op.name    = 'Scalar'
    Op.symbol  = str(val)
    Op.formula = str(val)
        
    Op.set_action(action, 'LR')    
    return Op


#------------------------------------------------------
def scalar_pot(func, mesh = None):
    """Scalar-potential operator.
    
    Returns the scalar-potential operator associated with `func'. 
    The optional `mesh' parameter triggers the evaluation of the potential 
    on the mesh for better performace.
    
    """

    if mesh:
        funcM = func(mesh.points)
        def action(wf, **kwds):
            return funcM * wf.copy()
    else:    
        def action(wf, **kwds):
            r = wf.mesh.points
            return func(r) * wf.copy()

    Op = Operator()
    Op.name    = 'Scalar potential'
    Op.symbol  = func.__name__
    Op.formula = ''
    Op.info    = '\n'+inspect.getsource(func)
        
    Op.set_action(action, 'LR')    
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
            dh = kwds.get('exp_step', exp_step)
            wf1 = wf.copy()
            factor = 1
            for i in range(1, order + 1):
                factor *=  dh / i
                wf1 = Opin.apply(wf1, side, **kwds)
                wf = factor*wf1 + wf 
            return wf.copy()

        return action

        
    Op = Operator()
            
    Op.set_action(get_action('L'), 'L')    
    Op.set_action(get_action('R'), 'R') 

    Op.name    = 'Exponential'
    Op.symbol  = 'Exp'
    Op.formula = 'exp(%s)'%Opin.symbol
    Op.info    = '%s = %s'%(Opin.symbol, Opin.formula)
       
    return Op

#------------------------------------------------------
def hamiltonian(mesh, vext = None, **kwds):
    """Creates an Hamiltonian operator.
    
    Utility to create an Hamiltonian.
        
    """

    T = kinetic(mesh, **kwds)
    H = T
    
    Vext = None
    if vext:
        Vext = scalar_pot(vext, mesh)
        H += Vext

    H.name    = 'Hamiltonian'
    H.symbol  = 'H'        
    H.kinetic = T
    H.vext = Vext 

    return H
        
    
#------------------------------------------------------
def kinetic(mesh, **kwds):
    """Kinetic operator"""
           
    T = -0.5 * Laplacian(mesh, **kwds)
    
    # def action(wf, side = 'LR', **kwds):
    #      print "well well"
    #      return  -0.5 * T.raction(wf, side=side, **kwds)    
    #      
    # T.set_action(action, 'LR')  

    T.name    = 'Kinetic'
    T.symbol  = 'T'
    T.formula = '-1/2 \\nabla^2'
        
    return T


#------------------------------------------------------
class Gradient(Operator):
    """Gradient operator"""
    def __init__(self, mesh, **kwds):
        super(Gradient, self).__init__(**kwds)
        self.name    = 'Gradient'
        self.symbol  = '\\nabla'
        self.formula = 'd/dx'
        
        self.der = kwds.get('Der', Derivative(mesh, **kwds)) 
    
    def apply(self,wfin, side='LR', **kwds): 
        return self.der.perform(wfin, degree = 1)        
        
    def _write_details(self, indent = 0):
        super(Gradient,self)._write_details(indent = indent)
        self.der.write_info(indent = indent+1)
        


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
    
    def apply(self,wfin, side='LR', **kwds):
        return self.der.perform(wfin, degree = 2)
                
    def _write_details(self, indent = 0):        
        super(Laplacian,self)._write_details(indent)
        self.der.write_info(indent = indent+1)


#------------------------------------------------------
class Kinetic(Laplacian):
    """Kinetic operator"""
    def __init__(self, mesh, **kwds):
        super(Kinetic,self).__init__(mesh, **kwds)
        self.name    = 'Kinetic'
        self.symbol  = 'T'
        self.formula = '-1/2 \\nabla^2'
    
    
    def apply(self, wfin, side='LR', **kwds):
        return -0.5*super(Kinetic,self).apply(wfin, side = side, **kwds)    
    


