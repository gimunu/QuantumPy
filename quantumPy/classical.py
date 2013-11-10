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

__all__ = ['PointParticle', 'Propagator']


import numpy as np
import copy


class PointParticle(object):
    """Classical PointParticle."""
    def __init__(self, **kwds):
        super(PointParticle, self).__init__()
        # The total number of degrees of freedom
        self.dim   = kwds.get('dim', 1)

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


    def __str__(self):
        return "Particle <M=%s, Q=%s; pos=%s, vel=%s>"%(self.mass, self.charge, self.currentPos, self.velocity)



class Propagator(object):
    """Classical time propagator"""
    def __init__(self, **kwds):
        super(Propagator, self).__init__()
        self.dt       = kwds.get('dt', 0.1)
        # self.forces   = kwds.get('forces', Forces())
        self.time = 0.0    
        
    def verlet(self, points):
        # Verlet integration step:
        for p in points:
            if not p.locked:
                # make a copy of our current position
                temp = p.currentPos.copy()
                p.currentPos += p.currentPos - p.oldPos + p.forces * self.dt**2
                p.oldPos = temp
    
    def accumulateForces(self, particles, **kwds):
        particles[:].p.forces = 0.0
        
        for p in particles:
            for F in self.forces:
                p.forces += F.evaluate(p, particles, time)
    
    def apply(self, particles, *kdws):
        self.dt   = get.kdws('dt'  , self.dt)
        self.time = get.kdws('time', self.time)
        
        self.accumulateForces(particles, time)
        self.verlet()
        for i in range(ITERATE):
            self.satisfyConstraints()


class Force(object):
    """docstring for Force"""
    def __init__(self, arg):
        super(Force, self).__init__()
        self.arg = arg

class Forces(object):
    """docstring for Forces"""
    def __init__(self, arg):
        super(Forces, self).__init__()
        self.arg = arg
        
                    