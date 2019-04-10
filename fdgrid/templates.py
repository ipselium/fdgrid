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


from .domain import Domains, Subdomain


class TemplateConstructionError(Exception):
    """ Bad value for grid template """
    pass


def testcase1(nx, nz):
    """ Test case with complex geometry. """

    geo = [Subdomain([0, 0, 60, 40], 'RRRR'),
           Subdomain([23, 40, 33, 50], 'RRRR'),
           Subdomain([56, 40, 60, 60], 'RRRR'),
           Subdomain([100, 80, 120, 90], 'RRRR'),
           Subdomain([90, 20, 110, 30], 'RRRR'),
           Subdomain([nx-90, nz-90, nx-60, nz-1], 'RRRR'),
           Subdomain([nx-60, nz-11, nx-1, nz-1], 'RRRR'),
           Subdomain([nx-60, nz-44, nx-30, nz-34], 'RRRR'),
           Subdomain([nx-60, nz-80, nx-40, nz-67], 'RRRR')]

    return Domains((nx, nz), data=geo)


def testcase2(nx, nz):
    """ Test case for periodic bc. """

    geo = [Subdomain([0, 0, 10, 10], 'RRRR'),
           Subdomain([0, 30, 10, int(3*nz/4)-5], 'RRRR'),
           Subdomain([30, nz-10, int(nx/2)+10, nz-1], 'RRRR'),
           Subdomain([int(nx/2), 0, int(3*nx/4), 10], 'RRRR'),
           Subdomain([nx-10, int(3*nz/4), nx-1, int(3*nz/4)+10], 'RRRR')]

    return Domains((nx, nz), data=geo)


def helmholtz(nx, nz, cavity=(0.2, 0.2), neck=(0.1, 0.1)):
    """ Helmholtz resonator.

    Parameters:
    -----------

    cavity (tuple): Normalized (width, height) of the cavity
    neck (tuple): Normalized (width, height) of the neck

    """

    neck_wdth = int(nx*neck[0])
    cvty_wdth = int(nx*cavity[0])
    neck_hght = int(nz*neck[1])
    cvty_hght = int(nz*cavity[1])

    neck_ix = int((nx - neck_wdth)/2)
    cvty_ix = int((nx - cvty_wdth)/2)


    if cavity[0] + neck[0] > 0.98 or cavity[1] + neck[1] > 0.98:
        raise TemplateConstructionError("resonator must be smaller than the domain")

    geo = [Subdomain([0, 0, cvty_ix, cvty_hght], 'RRRR'),
           Subdomain([cvty_ix+cvty_wdth, 0, nx-1, cvty_hght], 'RRRR'),
           Subdomain([0, cvty_hght, neck_ix, cvty_hght+neck_hght], 'RRRR'),
           Subdomain([neck_ix+neck_wdth, cvty_hght, nx-1, cvty_hght+neck_hght], 'RRRR')]

    return Domains((nx, nz), data=geo)


def helmholtz_double(nx, nz, cavity=(0.2, 0.2), neck=(0.1, 0.1)):
    """ Helmholtz resonator.

    Parameters:
    -----------

    cavity (tuple): Normalized (width, height) of the cavity
    neck (tuple): Normalized (width, height) of the neck

    """

    xneck_wdth = int(nx*neck[0])
    xcvty_wdth = int(nx*cavity[0])
    xneck_hght = int(nz*neck[1])
    xcvty_hght = int(nz*cavity[1])

    neck_ix = int((nx - xneck_wdth)/2)
    cvty_ix = int((nx - xcvty_wdth)/2)

    zneck_wdth = int(nz*neck[0])
    zcvty_wdth = int(nz*cavity[0])
    zneck_hght = int(nx*neck[1])
    zcvty_hght = int(nx*cavity[1])

    neck_iz = int((nz - zneck_wdth)/2)
    cvty_iz = int((nz - zcvty_wdth)/2)


    if cavity[0] + neck[0] > 0.98 or cavity[1] + neck[1] > 0.98:
        raise TemplateConstructionError("resonator must be smaller than the domain")

    geo = [Subdomain([0, 0, cvty_ix, xcvty_hght], 'RRRR'),
           Subdomain([cvty_ix+xcvty_wdth, 0, nx-1, xcvty_hght], 'RRRR'),
           Subdomain([0, xcvty_hght, neck_ix, xcvty_hght+xneck_hght], 'RRRR'),
           Subdomain([neck_ix+xneck_wdth, xcvty_hght, nx-1, xcvty_hght+xneck_hght], 'RRRR'),
           Subdomain([nx-zcvty_hght, xcvty_hght+xneck_hght, nx-1, cvty_iz], 'RRRR'),
           Subdomain([nx-zcvty_hght, cvty_iz+zcvty_wdth, nx-1, nz-1], 'RRRR'),
           Subdomain([nx-zcvty_hght-zneck_hght, xcvty_hght+xneck_hght, nx-zcvty_hght , neck_iz], 'RRRR'),
           Subdomain([nx-zcvty_hght-zneck_hght, neck_iz+zneck_wdth, nx-zcvty_hght, nz-1], 'RRRR')
            ]

    return Domains((nx, nz), data=geo)


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

    ixstart = int(ix0 - 1.5*size)
    izstart = int(iz0 - 0.5*size)

    geo = [Subdomain([ixstart, izstart, ixstart+size, izstart+size], 'RRRR'),
           Subdomain([ixstart+2*size, izstart, ixstart+3*size, izstart+size], 'RRRR'),
           Subdomain([ixstart+size, izstart-size, ixstart+2*size, izstart], 'RRRR'),
           Subdomain([ixstart+size, izstart+size, ixstart+2*size, izstart+2*size], 'RRRR')]

    return Domains((nx, nz), data=geo)


def square(nx, nz, size_percent=20):
    """ Square in the middle.

    Parameters:
    -----------

    size_percent (float): size of the square in percent of the largest
    dimension of the domain.
    """

    size = int(min(nx, nz)*size_percent/100)
    geo = [Subdomain([int(nx/2)-size, int(nz/2)-size,
                      int(nx/2)+size, int(nz/2)+size], 'RRRR')]

    return Domains((nx, nz), data=geo)


def street(nx, nz, street_size=50):
    """ Street with building facades. """

    geo = [Subdomain([0, 0, int(0.7*nx), int(nz*0.25)], 'RRRR'),
           Subdomain([int(0.8*nx), 0, nx-1, int(nz*0.25)], 'RRRR'),
           Subdomain([0, int(nz*0.75), nx-1, nz-1], 'RRRR'),
           Subdomain([int(0.05*nx), int(nz*0.7), int(0.15*nx), int(0.75*nz)], 'RRRR'),
           Subdomain([int(0.35*nx), int(nz*0.69), int(0.50*nx), int(0.75*nz)], 'RRRR'),
           Subdomain([int(0.60*nx), int(nz*0.72), int(0.70*nx), int(0.75*nz)], 'RRRR'),
           Subdomain([int(0.80*nx), int(nz*0.73), int(0.95*nx), int(0.75*nz)], 'RRRR'),
           Subdomain([int(0.80*nx), int(nz*0.25), int(0.95*nx), int(0.30*nz)], 'RRRR'),
           Subdomain([int(0.10*nx), int(nz*0.25), int(0.15*nx), int(0.30*nz)], 'RRRR'),
           Subdomain([int(0.30*nx), int(nz*0.25), int(0.38*nx), int(0.28*nz)], 'RRRR'),
           Subdomain([int(0.55*nx), int(nz*0.25), int(0.70*nx), int(0.28*nz)], 'RRRR')]

    return Domains((nx, nz), data=geo)
