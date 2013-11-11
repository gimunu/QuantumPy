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
sb = qp.classical.SimulationBox(dim=dim, size = [10.0]*dim)

electron = qp.classical.PointParticle(sbox = sb, velocity = [0., 1.5], position = [2.0, 0.0], charge = -1.0)

print electron

# F = qp.classical.constant_force(sb, [-0.5,-0.5])
proton = qp.classical.PointParticle(sbox =sb , charge = 1.0, locked = True, position = [0.0]*dim )
# F = qp.classical.electrostatic_force(sb, proton)
F = qp.classical.harmonic_force(sb, proton, damping = 0.0)
# F = qp.classical.Force(sb)
F.write_info()

anim = True

e1 = qp.classical.PointParticle(sbox = sb, velocity = [0., 2.5], position = [2.0, 0.0], charge = -1.0)
e1.record_start()

if anim:
    pl.ion()
    line, = pl.plot(electron.currentPos[0], electron.currentPos[1], 'bo', label='e')
    line1,  = pl.plot(e1.currentPos[0], e1.currentPos[1], 'go', label='e1')
    line1_, = pl.plot(e1.trajectory[0,0], e1.trajectory[0,1],  label='t1')
    pl.plot(proton.currentPos[0], proton.currentPos[1], 'ro', label='p')
    line.axes.set_xlim(-10,10)
    line.axes.set_ylim(-5,5)
         
    pl.legend( loc='upper left')
    pl.draw()

dt = 0.1
final_time = 50

U =  qp.classical.Propagator(F, dt = dt, method = 'verlet')
U.write_info()

U.initialize([electron])

U1 =  qp.classical.Propagator(F, dt = dt, method = 'velverlet')
U1.initialize([e1])

for i in range(0, int(final_time/dt)):
    time = i*dt
    U.apply([electron], time = time)
    U1.apply([e1], time = time)
    T = electron.kinetic_energy()

    print time, T, electron.velocity[:], electron.currentPos[:], electron.forces
    if anim:            
        line.set_xdata(electron.currentPos[0])
        line.set_ydata(electron.currentPos[1])
        line1.set_xdata(e1.currentPos[0])
        line1.set_ydata(e1.currentPos[1])
        line1_.set_xdata(e1.trajectory[:,0])
        line1_.set_ydata(e1.trajectory[:,1])
        pl.draw()
