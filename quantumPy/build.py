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

"""Build the physics engines for different theories.
   This module is a stub
"""

from __future__ import division

__all__ = ['World']


class World(object):
    """docstring for World"""
    def __init__(self, arg):
        super(World, self).__init__()
        self.arg = arg
        
class ShroedingerWorld(World):
    """docstring for ShroedingerWorld"""
    def __init__(self, arg):
        super(ShroedingerWorld, self).__init__()
        self.arg = arg

class NewtonWorld(World):
    """docstring for NewtonWorld"""
    def __init__(self, arg):
        super(NewtonWorld, self).__init__()
        self.arg = arg
                        
class MaxwellWorld(World):
    """docstring for MaxwellWorld"""
    def __init__(self, arg):
        super(MaxwellWorld, self).__init__()
        self.arg = arg
                