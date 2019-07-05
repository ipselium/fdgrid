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
#pylint: disable=too-many-locals
"""
-----------

The `templates` module provides a collection of examples to :

* build arangments of obstacles
* set curvilinear transformations for :py:class:`fdgrid.mesh.CurvilinearMesh`

-----------
"""

import numpy as _np
from fdgrid import _exceptions, Domain, Obstacle


#-----------------------------------------------------------------------------#
# Curvilinear transformations                                                 #
#-----------------------------------------------------------------------------#


def curv(xn, zn):
    """ Curvilinear coordinates: test case 1. Physical == numerical. """

    return xn.copy(), zn.copy()


def curvx(xn, zn):
    """ Curvilinear coordinates: test case 2. Following x. """

    zp = zn.copy()
    xp = xn \
        + _np.linspace(0.5, 0, xn.shape[1])*(_np.sin(2*_np.pi*zp/(zp.max()/10))/5000 \
                                            - 5*zp**2)
    return xp, zp


def curvz(xn, zn):
    """ Curvilinear coordinates: test case 2. Following z. """

    xp = xn.copy()
    zp = zn \
        + _np.linspace(0.5, 0, zn.shape[1])*(_np.sin(2*_np.pi*xp/(xp.max()/10))/5000 \
                                            - 5*xp**2)
    return xp, zp


def circle(xn, zn):
    """ Curvilinear coordinates: test case 3. The circle. """

    width = xn[-1, 0] - xn[0, 0]
    dx = xn[1, 0] - x[0, 0]
    R = (width+dx)/(2*_np.pi)
    xp = (zn + R)*_np.sin(xn/R)
    zp = (zn + R)*_np.cos(xn/R)

    return xp, zp


def section(xn, zn, variation=25, zmin=5, zmax=-5):
    """ Curvilinear coordinates: test case 4. The section change. """

    xp = xn.copy()
    if zmin < zmax:
        zp = zn \
            + _np.linspace(zmin, zmax, zn.shape[1]) \
            * (_np.pi/2 + _np.arctan(variation*_np.pi*xn/xn.max()))
    else:
        zp = zn \
            + _np.linspace(zmin, zmax, zn.shape[1]) \
            * (-_np.pi/2 + _np.arctan(variation*_np.pi*xn/xn.max()))

    return xp, zp

#-----------------------------------------------------------------------------#
# Geometry                                                                    #
#-----------------------------------------------------------------------------#

def testcase1(nx, nz):
    """ Geometry: Test case 1. Complex geometry. """

    geo = [Obstacle([0, 0, 60, 40], 'WWWW'),
           Obstacle([26, 40, 33, 50], 'WWWW'),
           Obstacle([56, 40, 60, 60], 'WWWW'),
           Obstacle([100, 80, 120, 90], 'WWWW'),
           Obstacle([90, 26, 110, 36], 'WWWW'),
           Obstacle([nx-90, nz-90, nx-60, nz-1], 'WWWW'),
           Obstacle([nx-60, nz-17, nx-1, nz-1], 'WWWW'),
           Obstacle([nx-60, nz-44, nx-30, nz-40], 'WWWW'),
           Obstacle([nx-60, nz-80, nx-40, nz-67], 'WWWW')]

    return Domain((nx, nz), data=geo)


def testcase2(nx, nz):
    """ Geometry: Test case 2. Periodic bc. """

    PML = 16

    geo = [Obstacle([0, 0, PML, PML], 'WWWW'),
           Obstacle([0, PML+23, PML, int(3*nz/4)-5], 'WWWW'),
           Obstacle([30, nz-PML-1, int(nx/2)+10, nz-1], 'WWWW'),
           Obstacle([int(nx/2), 0, int(3*nx/4), PML], 'WWWW'),
           Obstacle([nx-PML-1, int(3*nz/4), nx-1, int(3*nz/4)+10], 'WWWW')]

    return Domain((nx, nz), data=geo)


def helmholtz(nx, nz, cavity=(0.2, 0.2), neck=(0.1, 0.1)):
    """ Geometry: Helmholtz resonator.

    Parameters
    ----------
    nx : int
        Width of the domain.
    nz : int
        Height of the domain.
    cavity : tuple, optional
        Normalized (width, height) of the cavity.
    neck : tuple, optional
        Normalized (width, height) of the neck.

    """

    neck_wdth = int(nx*neck[0])
    cvty_wdth = int(nx*cavity[0])
    neck_hght = int(nz*neck[1])
    cvty_hght = int(nz*cavity[1])

    neck_ix = int((nx - neck_wdth)/2)
    cvty_ix = int((nx - cvty_wdth)/2)


    if cavity[0] + neck[0] > 0.98 or cavity[1] + neck[1] > 0.98:
        raise _exceptions.TemplateConstructionError("resonator must be smaller than the domain")

    geo = [Obstacle([0, 0, cvty_ix, cvty_hght], 'WWWW'),
           Obstacle([cvty_ix+cvty_wdth, 0, nx-1, cvty_hght], 'WWWW'),
           Obstacle([0, cvty_hght, neck_ix, cvty_hght+neck_hght], 'WWWW'),
           Obstacle([neck_ix+neck_wdth, cvty_hght, nx-1, cvty_hght+neck_hght], 'WWWW')]

    return Domain((nx, nz), data=geo)


def helmholtz_double(nx, nz, cavity=(0.2, 0.2), neck=(0.1, 0.1)):
    """ Geometry: Double Helmholtz resonator.

    Parameters
    ----------
    nx : int
        Width of the domain.
    nz : int
        Height of the domain.
    cavity : tuple, optional
        Normalized (width, height) of the cavity.
    neck : tuple, optional
        Normalized (width, height) of the neck.

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
        raise _exceptions.TemplateConstructionError("resonator must be smaller than the domain")

    geo = [Obstacle([0, 0, cvty_ix, xcvty_hght], 'WWWW'),
           Obstacle([cvty_ix+xcvty_wdth, 0, nx-1, xcvty_hght], 'WWWW'),
           Obstacle([0, xcvty_hght, neck_ix, xcvty_hght+xneck_hght], 'WWWW'),
           Obstacle([neck_ix+xneck_wdth, xcvty_hght, nx-1, xcvty_hght+xneck_hght], 'WWWW'),
           Obstacle([nx-zcvty_hght, xcvty_hght+xneck_hght, nx-1, cvty_iz], 'WWWW'),
           Obstacle([nx-zcvty_hght, cvty_iz+zcvty_wdth, nx-1, nz-1], 'WWWW'),
           Obstacle([nx-zcvty_hght-zneck_hght, neck_iz+zneck_wdth, nx-zcvty_hght, nz-1], 'WWWW'),
           Obstacle([nx-zcvty_hght-zneck_hght, xcvty_hght+xneck_hght,
                     nx-zcvty_hght, neck_iz], 'WWWW')
          ]

    return Domain((nx, nz), data=geo)


def plus(nx, nz, ix0=None, iz0=None, size=20):
    """ Geometry: Plus sign.

    Parameters
    ----------
    nx : int
        Width of the domain.
    nz : int
        Height of the domain.
    ix0 : int, optional
        x-location of the pattern.
    iz0 : int, optional
        z-location of the pattern.
    size : int
        Width of the pattern.
    """

    if not ix0:
        ix0 = nx/2

    if not iz0:
        iz0 = nz/2

    if ix0 <= 1.5*size or iz0 <= 0.5*size:
        msg = "Center of the plus must be greater than 1.5 time the size of a square"
        raise _exceptions.TemplateConstructionError(msg)

    ixstart = int(ix0 - 1.5*size)
    izstart = int(iz0 - 0.5*size)

    geo = [Obstacle([ixstart, izstart, ixstart+size, izstart+size], 'WWWW'),
           Obstacle([ixstart+2*size, izstart, ixstart+3*size, izstart+size], 'WWWW'),
           Obstacle([ixstart+size, izstart-size, ixstart+2*size, izstart], 'WWWW'),
           Obstacle([ixstart+size, izstart+size, ixstart+2*size, izstart+2*size], 'WWWW')]

    return Domain((nx, nz), data=geo)


def square(nx, nz, size_percent=20):
    """ Geometry: Square at the center of the domain.

    Parameters
    ----------
    nx : int
        Width of the domain.
    nz : int
        Height of the domain.
    size_percent : float
        size of the square (percent of the largest dimension of the domain).
    """

    size = int(min(nx, nz)*size_percent/100)
    geo = [Obstacle([int(nx/2)-size, int(nz/2)-size,
                     int(nx/2)+size, int(nz/2)+size], 'WWWW')]

    return Domain((nx, nz), data=geo)


def street(nx, nz):
    """ Geometry: Street model.

    Parameters
    ----------
    nx : int
        Width of the domain.
    nz : int
        Height of the domain.
    """

    geo = [Obstacle([0, 0, int(0.7*nx), int(nz*0.25)], 'WWWW'),
           Obstacle([int(0.8*nx), 0, nx-1, int(nz*0.25)], 'WWWW'),
           Obstacle([0, int(nz*0.75), nx-1, nz-1], 'WWWW'),
           Obstacle([int(0.11*nx), int(nz*0.7), int(0.15*nx), int(0.75*nz)], 'WWWW'),
           Obstacle([int(0.35*nx), int(nz*0.69), int(0.50*nx), int(0.75*nz)], 'WWWW'),
           Obstacle([int(0.60*nx), int(nz*0.72), int(0.70*nx), int(0.75*nz)], 'WWWW'),
           Obstacle([int(0.80*nx), int(nz*0.73), int(0.89*nx), int(0.75*nz)], 'WWWW'),
           Obstacle([int(0.80*nx), int(nz*0.25), int(0.89*nx), int(0.30*nz)], 'WWWW'),
           Obstacle([int(0.13*nx), int(nz*0.25), int(0.20*nx), int(0.30*nz)], 'WWWW'),
           Obstacle([int(0.30*nx), int(nz*0.25), int(0.38*nx), int(0.28*nz)], 'WWWW'),
           Obstacle([int(0.55*nx), int(nz*0.25), int(0.70*nx), int(0.28*nz)], 'WWWW')]

    return Domain((nx, nz), data=geo)


def moving_square(nx, nz, size_percent=20):
    """ Geometry: Square with moving bc.

    Parameters
    ----------
    nx : int
        Width of the domain.
    nz : int
        Height of the domain.
    size_percent : float
        size of the square (percent of the largest dimension of the domain).
    """

    size = int(min(nx, nz)*size_percent/100)

    obs1 = Obstacle([int(nx/2)-size, int(nz/2)-size,
                     int(nx/2)+size, int(nz/2)+size], 'VVWV')
    obs2 = Obstacle([nx-11, 0, nx-1, nz-1], 'VWWW')

    obs1.set_moving_bc({'f': 70000, 'A': 1, 'func': 'sine'},
                       {'f': 30000, 'A': -1, 'func': 'tukey'},
                       {'f': 30000, 'A': 1, 'func': 'tukey'})
    obs2.set_moving_bc({'f': 73000, 'A': -1, 'func': 'flat'})

    return Domain((nx, nz), data=[obs1, obs2])


def test_all():
    return [testcase1, plus, square]
