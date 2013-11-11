#!/usr/bin/env python

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
import pylab as pl


import quantumPy as qp

dim = 2
sb = qp.classical.SimulationBox(dim=dim)

electron = qp.classical.PointParticle(sbox = sb, velocity = [1.0]*dim, position = [2.0]*dim, charge = -1.0)
print electron

F = qp.classical.Force(sb)
F.write_info()

dt = 0.1
final_time = 10

U =  qp.classical.Propagator([F], dt = dt)

U.initialize([electron])
print  electron.velocity[:], electron.currentPos[:],  electron.oldPos[:]
for i in range(0, int(final_time/dt)):
    time = i*dt
    U.apply([electron], time = time)
    T = electron.kinetic_energy()

    print time, T, electron.velocity[:], electron.currentPos[:],  electron.oldPos[:]

