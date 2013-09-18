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

__all__ = ['base','grid','operators','td']

for module in __all__:
    exec 'from . import {0}'.format(module)

# Make selected functionality available directly in the root namespace.
available = [('operators', operators.__all__),
             ('grid.mesh', ['Box']), 
             ('td', ['Propagator'])]
for module, names in available:
    exec 'from .{0} import {1}'.format(module, ', '.join(names))
    __all__.extend(names)