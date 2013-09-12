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


printmsg = messages.print_msg  #shorter name     


#############################################
#
#############################################
class Operator(object):
    """Generic operator class acting on MeshFunction."""
    def __init__(self, **kwds):
        super(Operator, self).__init__()        
        self.name   = kwds.get('name', "Operator")
        self.symbol = kwds.get('symbol', "Op")
        self.formula = kwds.get('formula', None)
        self.hermitian = kwds.get('hermitian', True)

    def apply(self,wfin):
        """ Apply the operator to a wavefunction."""
        return  self.applyRigth(wfin)

    def expectationValue(self, wfR, wfL = None):
        """ Calculate the expectation value of the operator"""
        if wfL == None:
            wfL = wfR.copy().conjugate()

        out = (wfL * self.applyRigth(wfR)).integrate()

        return out 
        
        
    def applyRigth(self,wfinR):    
        # User should overload this method to define the operator (right) action    
        raise Exception( "Error: applyRigth() method of %s is Undefined."%(self.name))
        return wfout

    def applyLeft(self,wfinL):    
        # apply the operator
        raise Exception( "Error: applyLeft() method of %s is Undefined."%(self.name)) 
        return wfout

    def write_info(self, indent = 0):
        from functools import partial
        printmsg = partial(messages.print_msg, indent = indent)       
        print_msg = messages.print_msg    
        
        printmsg( "%s operator (%s): "%(self.name, self.symbol) )       
        if self.formula != None:
            print_msg( "%s = %s "%(self.symbol, self.formula), indent = indent+1)    


#############################################
#
#############################################
class Identity(Operator):
    """Identity operator"""
    def __init__(self, **kwds):
        super(Identity, self).__init__(**kwds)
        self.name    = 'Identity'
        self.symbol  = 'I'
        self.formula = '1'
        
    def applyRigth(self, wfinR): 
        return wfinR

    def applyLeft(self, wfinL): 
        return wfinL


#############################################
#
#############################################
class Laplacian(Operator):
    """Laplacian operator"""
    def __init__(self, mesh, **kwds):
        super(Laplacian, self).__init__(**kwds)
        self.mesh = mesh
        
        self.name    = 'Laplacian'
        self.symbol  = '\Nabla^2'
        self.formula = 'd^2/dx^2'
        if mesh.dim > 1:
            self.formula += '+ d^2/dy^2' 
        if mesh.dim > 2:
            self.formula += '+ d^2/dz^2' 
        
        self.strategy = kwds.get('Strategy', 'finite difference')
        
        self.cube = None # A cube mesh to perform FFTs
        if (self.strategy.lower() == 'fourier'):
            self.cube = Cube(self.mesh, Attributes = 'RS + FS')

    def write_info(self, indent = 0):
        from functools import partial
        printmsg = partial(messages.print_msg, indent = indent)       
        print_msg = messages.print_msg       
        
        super(Laplacian,self).write_info(indent)
        print_msg( "strategy  = %s "%(self.strategy), indent = indent+1)
        
        if (self.cube != None):
            self.cube.write_info(indent = indent + 1)
                
    def applyRigth(self,wfinR):
        
        wfout = wfinR.copy()
        
        if   self.strategy.lower() == 'finite difference':
            #do something
            wfout = wfinR
            
        elif self.strategy.lower() == 'fourier': 
            # Use derivatives in Fourier space F(f'(x))=ik f(k), 
            # hence:
            # \Nabla^2 = F^{-1}(-k^2 F(f(x)))
            
            # self.cube = self.cube if (self.cube != None) else Cube(wfinR.mesh) # Create a new cube mesh if needed
            # self.cube.write_info()

            cf = mesh_to_cube(wfinR, self.cube)
            Fcf = rs_to_fs(cf)
            K = Fcf.mesh.FSpoints        
            cf = fs_to_rs(K**2.0 * Fcf)
            wfout = cube_to_mesh(cf, wfinR.mesh)
            
        else:
            raise Exception("Unrecognized option")
         
        return wfout
         
    def applyLeft(self,wfinL):
        return self.applyRigth(wfinL.conjugate())      
             



#############################################
#
#############################################
class Kinetic(Laplacian):
    """Kinetic operator"""
    def __init__(self, mesh, **kwds):
        super(Kinetic,self).__init__(mesh, **kwds)
        self.name    = 'Kinetic'
        self.symbol  = 'T'
        self.formula = '1/2 \Nabla^2'
        
    def applyRigth(self,wfinR): 
        return 0.5*super(Kinetic,self).applyRigth(wfinR)
    
        

# def Kinetic(psiG, mesh):
#     kN   = mesh.fs.points
#     out = kN**2 * psiG / 2.0
#     return out
#     
# 
# def Vext(psiG, mesh):
#     psiGx  = fs_to_rs(psiG)
#     vextGx  = VextF(mesh.rs.points)
#     vpsiGx =  vextGx * psiGx
#     out = rs_to_fs(vpsiGx)
#     return out
# 
# def H(psiG, mesh):
#     Hpsi = Kinetic(psiG, mesh) + Vext(psiG, mesh) 
#     return Hpsi
# 
# def U(psiG, mesh, dt): 
#     HNpsi = psiG
#     Upsi = np.zeros(psiG.size) + 0j
#     N = 5 # order of the exponential
#     for i in range(N):
#         Upsi +=  HNpsi * (- 1j * dt)**i /factorial(i)
#         HNpsi = H(HNpsi, mesh)      
# 
#     return Upsi     