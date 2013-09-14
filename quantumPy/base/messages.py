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

import logging

#Constants
_QUIET  = False
_INDENT = 0 
_INDENT_TAB = 2 

_MSG_LINE_CHAR = "#"

# logging.basicConfig(level=logging.WARNING if quiet else logging.INFO,
#                     format="%(message)s")
logging.basicConfig(level= logging.INFO, format="%(message)s")


def print_msg(string, indent = _INDENT, line_char = _MSG_LINE_CHAR):
    """simple messages printing helpers"""
    if string:
        for lev in range(indent):
            for i in range(_INDENT_TAB):
                string = " "+string   
        string = line_char + " " + string
        print string
        # logging.info(string)

    
    