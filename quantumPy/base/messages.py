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

__all__=['print_msg', 'debug_msg']

import logging

#Public constants
DEBUG_LEVEL = 0

#Private constants
_QUIET  = False
_INDENT = 0 
_INDENT_TAB = 2 

_MSG_TRAIL = "#"
_DBG_TRAIL = "*"

# logging.basicConfig(level=logging.WARNING if quiet else logging.INFO,
#                     format="%(message)s")
logging.basicConfig(level= logging.INFO, format="%(message)s")

def debug_msg(string, indent = _INDENT, line_char = _DBG_TRAIL, lev = 0):
    if DEBUG_LEVEL > lev:
         print_msg(string, indent = indent, line_char = line_char)
    

def print_msg(string, indent = _INDENT, line_char = _MSG_TRAIL):
    """simple messages printing helpers"""
    if string:
        for st in filter(None, string.split('\n')):
            for lev in range(indent):
                for i in range(_INDENT_TAB):
                    st = " " + st   
            st = line_char + " " + st
            print st
        # logging.info(string)

    
    