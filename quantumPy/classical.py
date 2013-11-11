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
        self.size   = np.array([1.0]*self.dim)
        self.center = np.array([0.0]*self.dim)
        
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
        self.currentPos = np.array(kwds.get('position', [0]*self.dim))
        self.oldPos = self.currentPos.copy()

        self.velocity = np.array(kwds.get('velocity', [0]*self.dim))
        self.forces   = np.array(kwds.get('forces', [0]*self.dim))
        
        # Should the particle be locked at its current position?
        self.locked = np.array(kwds.get('locked', False))

        # Record the trajectory?
        self.record = False
        self.trajectory = np.array([])
        

    def __str__(self):
        return "Particle <M=%s, Q=%s; pos=%s, vel=%s>"%(self.mass, self.charge, self.currentPos, self.velocity)

    def record_start(self):
        self.record = True

    def record_stop(self):
        self.record = False

    def kinetic_energy(self):
        return np.dot(self.velocity, self.velocity)/(2.*self.mass)


class Propagator(object):
    """Classical time propagator"""
    def __init__(self, forces, **kwds):
        super(Propagator, self).__init__()
        self.forces   = forces
        self.dt       = kwds.get('dt', 0.1)
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
    
    def accumulateForces(self, **kwds):
        particles = kwds.get('particles', None)

        for p in particles:
            p.forces = np.array([0]*p.dim)
        
        for p in particles:
            for F in self.forces:
                p.forces += F.evaluate(p, **kwds)
    
    def apply(self, particles, **kwds):
        self.dt   = kwds.get('dt'  , self.dt)
        self.time = kwds.get('time', self.time)
        
        self.accumulateForces(particles = particles, time = self.time )
        self.verlet(particles)
        # for i in range(ITERATE):
        #     self.satisfyConstraints()

    def initialize(self, particles):
        for p in particles:
            p.oldPos = p.currentPos - p.velocity *self.dt
        


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

        self.forceField = None 
        
    def write_info(self, indent = 0): 
        print_msg( "%s force (%s): "%(self.name, self.symbol), indent = indent )       
        if self.formula:
            print_msg( "%s = %s "%(self.symbol, self.formula), indent = indent+1)    
        if self.info:
            print_msg( "info: %s "%(self.info), indent = indent+1)    
    
    def evaluate(self, particle, **kwds):
        
        F = [0]*self.dim
        
        if getattr(self, 'forceField'):
            forceField = getattr(self, forceField)
            F = forceField(particle.currentPos[:], **kwds)
        else:
           pass 
            # wfout = self._evaluate(self.expr, wfin, side, **kwds)
        
        return F  

    def __iter__(self):
        if self:
            for elem in self.forces:
                yield elem
                      