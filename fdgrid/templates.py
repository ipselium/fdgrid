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
#
# Creation Date : 2019-02-13 - 10:58:09
"""
-----------

Examples of obstacle arangements.

@author: Cyril Desjouy
"""


from .domain import Subdomain


class TemplateConstructionError(Exception):
    """ Bad value for grid template """
    pass


def random(nx, nz):
    """ Random arrangement of obstacles. """
    return [Subdomain([0, 0, 60, 40], 'RRRR'),
            Subdomain([23, 40, 33, 50], 'RRRR'),
            Subdomain([56, 40, 60, 60], 'RRRR'),
            Subdomain([100, 80, 120, 90], 'RRRR'),
            Subdomain([90, 20, 110, 30], 'RRRR'),
            Subdomain([nx-90, nz-90, nx-60, nz-1], 'RRRR'),
            Subdomain([nx-60, nz-11, nx-1, nz-1], 'RRRR'),
            Subdomain([nx-60, nz-44, nx-30, nz-34], 'RRRR'),
            Subdomain([nx-60, nz-80, nx-40, nz-67], 'RRRR')]


def helmholtz(nx, nz, cavity=(0.2, 0.2), neck=(0.1, 0.1)):
    """ Helmholtz resonator.

    Parameters:
    -----------

    cavity (tuple): Normalized (width, height) of the cavity
    neck (tuple): Normalized (width, height) of the neck

    """

    neck_width = int(nx*neck[0])
    cvty_width = int(nx*cavity[0])
    neck_height = int(nz*neck[1])
    cvty_height = int(nz*cavity[1])

    neck_ix = int((nx - neck_width)/2)
    cvty_ix = int((nx - cvty_width)/2)


    if cavity[0] + neck[0] > 0.98 or cavity[1] + neck[1] > 0.98:
        raise TemplateConstructionError("resonator must be smaller than the domain")

    return [Subdomain([0, 0, cvty_ix, cvty_height], 'RRRR'),
            Subdomain([cvty_ix+cvty_width, 0, nx-1, cvty_height], 'RRRR'),
            Subdomain([0, cvty_height, neck_ix, cvty_height+neck_height], 'RRRR'),
            Subdomain([neck_ix+neck_width, cvty_height, nx-1, cvty_height+neck_height], 'RRRR')]


def plus(nx, nz, ix0=None, iz0=None, size=20):
    """ Plus sign.

    Parameters:
    -----------

    size (int): size of a square (number of points)
    """

    if not ix0:
        ix0 = nx/2

    if not iz0:
        iz0 = nz/2

    if ix0 <= 1.5*size or iz0 <= 0.5*size:
        msg = "Center of the plus must be greater than 1.5 time the size of a square"
        raise TemplateConstructionError(msg)

    ix_start = int(ix0 - 1.5*size)
    iz_start = int(iz0 - 0.5*size)

    return [Subdomain([ix_start, iz_start, ix_start+size, iz_start+size], 'RRRR'),
            Subdomain([ix_start+2*size, iz_start, ix_start+3*size, iz_start+size], 'RRRR'),
            Subdomain([ix_start+size, iz_start-size, ix_start+2*size, iz_start], 'RRRR'),
            Subdomain([ix_start+size, iz_start+size, ix_start+2*size, iz_start+2*size], 'RRRR')]


def square(nx, nz, size_percent=20):
    """ Square in the middle.

    Parameters:
    -----------

    size_percent (float): size of the square in percent of the largest dimension of the domain.
    """

    size = int(min(nx, nz)*size_percent/100)

    return [Subdomain([int(nx/2)-size, int(nz/2)-size, int(nx/2)+size, int(nz/2)+size])]
