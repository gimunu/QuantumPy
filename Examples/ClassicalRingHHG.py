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

def classical_ring_hhg(**kwds):
    
    
    verbose = kwds.get('verbose', False)
    
    dim = 2
    sb = qp.classical.SimulationBox(dim=dim, size = [10.0]*dim)


    # omega = 1.55/27.211
    omega = 0.5
    # omega = 0.6297613001523616/27.211
    omega = kwds.get('omega', omega)
    tc = 2.*np.pi/omega
    nconst = 32
    nup = 2
    nconst = kwds.get('nconst', nconst)
    ntot = nconst + 2*nup
    tconst = nconst * tc 
    tramp = nup * tc
    tau = (tconst + 2.*tramp)/2.
    # A = 0.12
    I = kwds.get('Intensity', 4e+14)
    A = np.sqrt(I /3.5094704445145668e+16)
    envelope = qp.td.tdf_trapezoidal( A, tconst, tramp, tau)
    # envelope = qp.td.tdf_constant(A)
    pol = kwds.get('polarzation', [0,1])
    laser = qp.td.ExternalField(sb, omega = omega, envelope = envelope, polarization = pol)
    E = qp.classical.td_external_field(sb, externalfield = laser)


    # F.forces = [E]


    # Force an electron to move on a circle centered in ec
    Radius = kwds.get('radius', 2.7) 
    theta0 = kwds.get('theta0', 0.0)
    omega0 = kwds.get('omega0', 0.0)
    mass = kwds.get('mass', 1.0)
    
    ec = qp.classical.PointParticle(sbox = sb, locked = True, position = [0.0, 0.0], charge = 0.0)
    e  = qp.classical.PointParticle(sbox = sb, velocity = [0., 0], position = [Radius*np.cos(theta0),Radius*np.sin(theta0)], charge = -1.0, mass = mass)
    e.record_start()
    C = qp.classical.Constraint(ec, e)
    ring = qp.classical.ParticleSystem([ec, e], constraints = C)
    # ring = qp.classical.ParticleSystem([ec, e]) # just a free electron


    dt = kwds.get('dt', 0.05)
    final_time = tc*(ntot + 2)

    times = np.linspace(0.0, final_time, num = int(final_time/dt), endpoint=False)
    laser.write_info(times = times)
    print "# Quiver lenght = %1.4e [a.u]"%(-2.0*e.charge*A/(e.mass*omega**2.))


    anim = kwds.get('animation',False)

    if anim:
        pl.ion()
        fig = pl.figure()
        p1 = fig.add_subplot(2, 1, 1, title = "Real Time")
        line,  = p1.plot(e.pos[0], e.pos[1], 'go', label='ring')
        line_, = p1.plot(e.trajectory[0,0], e.trajectory[0,1],  label='t2')
        line.axes.set_xlim(-5,5)
        line.axes.set_ylim(-5,5)
        p1.legend( loc='upper left')
         
        p2 = fig.add_subplot(2, 1, 2, title = "laser")
        p2.plot(laser.time, laser.field,  label='laser')
        lpoint, = p2.plot(0, laser.field[0,1], 'ro')
        p2.plot(laser.time, [envelope.evaluate(t) for t in times],  label='envelope')

        pl.draw()


    U =  qp.classical.Propagator(E, dt = dt, method = 'verlet')
    U.write_info()

    U.initialize(ring)

    velocity = np.zeros((times.size, sb.dim)) 
    pos      = np.zeros((times.size, sb.dim)) 
    acc      = np.zeros((times.size, sb.dim)) 

    Nt  = int(final_time/dt) 
    for i in range(0, Nt):
        time = i*dt
        U.apply(ring, time = time)
        velocity[i] = e.velocity
        pos[i] = e.pos
        # acc[i] = velocity[i]-velocity[i-1] if (i > 0) else 0.0 
        # 2p fd 
        acc[i] = pos[i-1] + pos[i+1] -2 * pos[i] if (i > 1 and i < Nt -1) else 0.0 
        T = e.kinetic_energy()
        E = T
        R = ring.particles[1].pos -ring.particles[0].pos
        if verbose:
            print "%d\t %s\t %f\t %f -- %f"%(i, time, E, T ,  np.sqrt(np.dot(R,R))) 
        if anim:            
            line.set_xdata(e.pos[0])
            line.set_ydata(e.pos[1])
            line_.set_xdata(e.trajectory[:,0])
            line_.set_ydata(e.trajectory[:,1])
            lpoint.set_xdata(time)
            lpoint.set_ydata(laser.field[i,1])
            pl.draw()
    
    acc = acc/dt**2

    spect = np.zeros((times.size, sb.dim))+0j
    for i in range(sb.dim):
        spect[:,i] = np.fft.fftshift(np.fft.fft(acc[:,i]))

    w = np.fft.fftfreq(times.size, d =dt/(2.*np.pi)) 
    w = np.fft.fftshift(w)/omega

    spectx = np.zeros((times.size, sb.dim))+0j
    for i in range(sb.dim):
        spectx[:,i] = np.fft.fftshift(np.fft.fft(pos[:,i]))

    
    
    if kwds.get('plot',False):
        fig2 = pl.figure()

        p1 = fig2.add_subplot(4, 1, 1, title = "position")
        p1.plot(times, pos[:,0],  label='x')
        p1.plot(times, pos[:,1],  label='y')
        p1.legend( loc='lower left')

        p2 = fig2.add_subplot(4, 1, 2, title = "velocity")
        p2.plot(times, velocity[:,0],  label='vx')
        p2.plot(times, velocity[:,1],  label='vy')
        p2.legend( loc='lower left')

        p3 = fig2.add_subplot(4, 1, 3, title = "acceleration")
        p3.plot(times, acc[:,0],  label='ax')
        p3.plot(times, acc[:,1],  label='ay')
        p3.legend( loc='lower left')

        p4 = fig2.add_subplot(4, 1, 4, title = "Laser")
        p4.plot(times, laser.field[:,0],  label='Ex')
        p4.plot(times, laser.field[:,1],  label='Ey')
        p4.legend( loc='lower left')

        pl.tight_layout()

        fig3 = pl.figure()
        p = fig3.add_subplot(1, 1, 1, title = "Harmonic specrtum from acc (2nd order FD)")
        p.plot(w, np.abs(spect[:,0])**2+np.abs(spect[:,1])**2)
        # p.axes.set_xlim([0,np.amax(w)])
        p.axes.set_xlim([0,100])
        p.set_yscale('log')

        fig4 = pl.figure()
        p = fig4.add_subplot(1, 1, 1, title = "Harmonic specrtum from position")
        p.plot(w, np.abs(spectx[:,0])**2+np.abs(spectx[:,1])**2)
        # p.axes.set_xlim([0,np.amax(w)])
        p.axes.set_xlim([0,100])
        p.set_yscale('log')

        pl.show()
    
    return (times, pos, velocity, acc, laser.field,  w, spect, spectx)


#############    
# MAIN 
############
if __name__ == '__main__':

    omega = 0.6297613001523616/27.211

    times, pos, vel, acc, laser, w, spect, spectx= \
    classical_ring_hhg(omega = omega, nconst = 10, plot = True)


