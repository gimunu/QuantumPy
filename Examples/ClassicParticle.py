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

electron = qp.classical.PointParticle(sbox = sb, velocity = [0., 0.0], position = [2.0, 0.0], charge = -1.0)

print electron

proton = qp.classical.PointParticle(sbox =sb , charge = 1.0, locked = True, position = [0.0]*dim )
F = qp.classical.electrostatic_force(sb, proton, softcore = True)
# F = qp.classical.harmonic_force(sb, proton, damping = 0.0)
# F = qp.classical.Force(sb)
# F = qp.classical.constant_force(sb, [-0.5,0.0])
F.write_info()

omega = 1
tc = 2.*np.pi/omega
nconst = 28
nup = 2
tconst = nconst * tc 
tramp = nup * tc
tau = (tconst + 2.*tramp)/2.
A = 1
envelope = qp.td.tdf_trapezoidal( A, tconst, tramp, tau)
E = qp.classical.td_external_field(sb, omega = omega, envelope = envelope)

# F.forces = [F, E]
F = E

anim = True

e1 = qp.classical.PointParticle(sbox = sb, velocity = [0., 2.5], position = [2.0, 0.0], charge = -1.0)
e1.record_start()

# Force an electron to move on a circle centered in ec
ec = qp.classical.PointParticle(sbox = sb, locked = True, position = [1.0, 1.0])
e  = qp.classical.PointParticle(sbox = sb, velocity = [0., 2.5], position = [3.0, 0.0], charge = -1.0)
e.record_start()
C = qp.classical.Constraint(ec, e)
ring = qp.classical.ParticleSystem([ec, e], constraints = C)

if anim:
    pl.ion()
    line, = pl.plot(electron.pos[0], electron.pos[1], 'bo', label='e')
    line1,  = pl.plot(e1.pos[0], e1.pos[1], 'go', label='e1')
    line1_, = pl.plot(e1.trajectory[0,0], e1.trajectory[0,1],  label='t1')
    line2,  = pl.plot(e.pos[0], e.pos[1], 'go', label='ring')
    line2_, = pl.plot(e.trajectory[0,0], e.trajectory[0,1],  label='t2')
    pl.plot(proton.pos[0], proton.pos[1], 'ro', label='p')
    line.axes.set_xlim(-5,5)
    line.axes.set_ylim(-5,5)
         
    pl.legend( loc='upper left')
    pl.draw()

dt = 0.05
final_time = 50

U =  qp.classical.Propagator(F, dt = dt, method = 'velverlet')
U.write_info()

U.initialize([electron])
U.initialize(ring)

U1 =  qp.classical.Propagator(F, dt = dt, method = 'verlet')
U1.initialize([e1])

for i in range(0, int(final_time/dt)):
    time = i*dt
    U.apply([electron], time = time)
    U.apply(ring, time = time)
    U1.apply([e1], time = time)
    T = electron.kinetic_energy()
    V = F.evaluate(electron, quantity = 'potential')
    E = T+V
    R = ring.particles[1].pos -ring.particles[0].pos
    print "%d\t %s\t %f\t %f\t %f -- %f"%(i, time, E, T , V, np.dot(R,R)) 
    if anim:            
        line.set_xdata(electron.pos[0])
        line.set_ydata(electron.pos[1])
        line1.set_xdata(e1.pos[0])
        line1.set_ydata(e1.pos[1])
        line1_.set_xdata(e1.trajectory[:,0])
        line1_.set_ydata(e1.trajectory[:,1])
        line2.set_xdata(e.pos[0])
        line2.set_ydata(e.pos[1])
        line2_.set_xdata(e.trajectory[:,0])
        line2_.set_ydata(e.trajectory[:,1])
        pl.draw()
