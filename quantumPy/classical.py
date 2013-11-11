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

"""Routines to compute classical dynamics"""

from __future__ import division

__all__ = ['PointParticle', 'Propagator', 'Force']


import numpy as np
import copy
from .base import *
from .grid import *


class SimulationBox(object):
    """docstring for SimulationBox"""
    def __init__(self, **kwds):
        super(SimulationBox, self).__init__()
        self.dim = kwds.get('dim', 1)
        self.size   = np.array(kwds.get('size',   [1.0]*self.dim))
        self.center = np.array(kwds.get('center', [0.0]*self.dim))
        
    def write_info():
        pass
            

class PointParticle(object):
    """Classical PointParticle."""
    def __init__(self, sbox, **kwds):
        super(PointParticle, self).__init__()
        # The total number of degrees of freedom
        assert(isinstance(sbox, SimulationBox))
        self.sb    = sbox
        self.dim   = self.sb.dim

        # Physical properties
        self.mass   = kwds.get('mass', 1.0)
        self.charge = kwds.get('charge', 1.0)

        # Physical variables 
        self.currentPos = np.array(kwds.get('position', [0.0]*self.dim))
        self.oldPos = self.currentPos.copy()

        self.velocity = np.array(kwds.get('velocity', [0.0]*self.dim))
        self.forces   = np.array(kwds.get('forces', [0.0]*self.dim))
        
        # Should the particle be locked at its current position?
        self.locked = np.array(kwds.get('locked', False))

        # Record the trajectory?
        self.rec = False
        self.trajectory = np.array([])
        

    def __str__(self):
        return "Particle <M=%s, Q=%s; pos=%s, vel=%s>"%(self.mass, self.charge, self.currentPos, self.velocity)

    def record(self):
        self.trajectory=np.append(self.trajectory, [self.currentPos], axis = 0)

    def record_start(self):
        self.rec = True
        self.trajectory=np.array([self.currentPos])

    def record_stop(self):
        self.rec = False

    def kinetic_energy(self):
        return np.dot(self.velocity, self.velocity)/(2.*self.mass)


class Propagator(object):
    """Classical time propagator"""
    def __init__(self, forces, **kwds):
        super(Propagator, self).__init__()
        self.forces   = forces
        self.dt       = kwds.get('dt', 0.1)
        self.method    = kwds.get('method', 'velverlet')
        self.time = 0.0    
        
    def verlet(self, particles):
        # Verlet integration step:
        for p in particles:
            if not p.locked:
                # make a copy of our current position
                temp = p.currentPos.copy()
                p.currentPos = 2.*p.currentPos - p.oldPos + p.forces/p.mass * self.dt**2.
                p.oldPos = temp
                # Approximated 
                p.velocity = (p.currentPos - p.oldPos)/self.dt 

    def velverlet(self, particles):
        # Velocity Verlet integration step:
        for p in particles:
            if not p.locked:
                # r(t+dt)
                p.currentPos = p.currentPos + p.velocity*self.dt + 0.5*p.forces/p.mass * self.dt**2.
                # v(t+dt/2)
                p.velocity = p.velocity + 0.5*p.forces/p.mass * self.dt
                # f(t+dt) 
                # note that if f(t+dt) depends on v (like with damping) we are making an error
                # since we at this step we only have access to v(t+dt/2)
                self.accumulateForces(particle = p)
                # v(t+dt)
                p.velocity = p.velocity + 0.5*p.forces/p.mass * self.dt

    
    def accumulateForces(self, **kwds):
        p = kwds.pop('particle', None)

        p.forces = np.array([0.0]*p.dim)        
        for F in self.forces:
            p.forces += F.evaluate(p, **kwds)

    def accumulateTotalForces(self, **kwds):
        particles = kwds.get('particles', None)
        
        for p in particles:
            self.accumulateForces(particle = p)
    
    def apply(self, particles, **kwds):
        self.dt   = kwds.get('dt'  , self.dt)
        self.time = kwds.get('time', self.time)
        
        self.accumulateTotalForces(particles = particles, time = self.time )
        
        if   self.method == 'verlet':
            self.verlet(particles)
        elif self.method == 'velverlet':
            self.velverlet(particles)
        else:
            raise Exception
        
        for p in particles:
            if p.rec:
                p.record()    
            
        # for i in range(ITERATE):
        #     self.satisfyConstraints()

    def initialize(self, particles):
        for p in particles:
            p.oldPos = p.currentPos - p.velocity *self.dt
            
    
    def write_info(self, indent = 0):
         print_msg( "Classical time propagator: ", indent = indent )          
         print_msg( "method = %s"%(self.method), indent = indent+1 )          
        


class Force(object):
    """Classical Force class"""
    def __init__(self, sbox, **kwds):
        super(Force, self).__init__()

        assert(isinstance(sbox, SimulationBox))
        self.sb    = sbox
        self.dim   = self.sb.dim
        
        
        self.name    = kwds.get('name'  , 'Force')
        self.symbol  = kwds.get('symbol' , "F")
        self.formula = kwds.get('formula', self.symbol)
        self.info    = kwds.get('info'   , None)
        

        self.forces  = [self]
        # self.expr    = BinaryTree(self)

        self.forcefield = None 
        
    def write_info(self, indent = 0): 
        print_msg( "%s force (%s): "%(self.name, self.symbol), indent = indent )       
        if self.formula:
            print_msg( "%s = %s "%(self.symbol, self.formula), indent = indent+1)    
        if self.info:
            print_msg( "info: %s "%(self.info), indent = indent+1)    
    
    def evaluate(self, particle, **kwds):
        
        F = [0.0]*self.dim
        
        if getattr(self, 'forcefield'):
            forcefield = getattr(self, 'forcefield')
            F = forcefield(particle, **kwds)
        # else:
        #    pass 
        #     # wfout = self._evaluate(self.expr, wfin, side, **kwds)
        
        return F  

    def set_forcefield(self, func):
        self.forcefield = func
                

    def __iter__(self):
        if self:
            for elem in self.forces:
                yield elem


#############################################
#  Library of Forces
#############################################

def null_force(sb, vec):

    def forcefield(p, **kwds):
        return [0.0]*sb.dim

    F = Force(sb)
    F.name    = 'Constant'
    F.symbol  = 'F'
    F.formula = '0'
        
    F.set_forcefield(forcefield)    
    return F


def constant_force(sb, vec):

    dimvec = len(vec)
    if dimvec<sb.dim:
        v = [0.0]*sb.dim
        v[0:dimvec] = vec[0:dimvec]
        vec = v 
    
    def forcefield(p, **kwds):
        return vec[0:sb.dim] #if (p.shape[0] >= np.array(vec).shape[0])

    F = Force(sb)
    F.name    = 'Constant'
    F.symbol  = 'F'
    F.formula = '%s'%(vec)
        
    F.set_forcefield(forcefield)    
    return F


def electrostatic_force(sb, point, const = 1.0):

    def forcefield(p, **kwds):
        R = p.currentPos - point.currentPos
        nR = np.dot(R,R)
        return const * point.charge *p.charge * R /nR**(3./2.)  

    F = Force(sb)
    F.name    = 'Electrostatic'
    F.symbol  = 'E'
    F.formula = 'C * Q * q r/||R-r||^2 \n C=%1.4e \n Q=%s \n R=%s'%(const, point.charge, point.currentPos)
        
    F.set_forcefield(forcefield)    
    return F


def harmonic_force(sb, point, k = 1.0, damping = 0.0):

    def forcefield(p, **kwds):
        R = p.currentPos - point.currentPos
        return -k * R  - damping * p.velocity

    F = Force(sb)
    F.name    = 'Spring'
    F.symbol  = 'Fs'
    F.formula = '-k * (r - R)\n C=%1.4e \n R=%s'%(k, point.currentPos)
        
    F.set_forcefield(forcefield)    
    return F

