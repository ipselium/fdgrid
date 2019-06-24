#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2016-2019 Cyril Desjouy <cyril.desjouy@univ-lemans.fr>
#
# This file is part of fdgrid
#
# fdgrid is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# fdgrid is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with fdgrid. If not, see <http://www.gnu.org/licenses/>.
#
# Creation Date : 2019-05-01 - 22:57:00
"""
-----------

Exceptions for fdgrid

-----------
"""

class CloseObstaclesError(Exception):
    """ Exception when obstacles are too close. """

    msg = "{} too close to another subdomain for a {} points stencil."


class BoundaryConditionError(Exception):
    """ Exception when incompatible boundary conditions """
    pass


class GridError(Exception):
    """ Exception when wrong grid parameters. """
    pass


class TemplateConstructionError(Exception):
    """ Bad value for grid template """
    pass
