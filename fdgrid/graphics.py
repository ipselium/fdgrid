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
# Creation Date : 2019-05-26 - 00:03:36
# pylint: disable=too-many-locals
"""
-----------
Figure setup functions


@author: Cyril Desjouy
"""

import numpy as _np
import matplotlib.patches as _patches
from fdgrid import utils as _utils, Subdomain


def plot_grid(ax, x, z, N=8):
    """ Plot grid lines.

    Parameters
    ----------

    ax : Matplotlib axe object where grid must be plotted
    x, z : x and z axes. Must be 1D or 2D ndarrays
    N : Plot 1 over N line of the gridmesh. Optional
    """

    if len(x.shape) == 1 and len(z.shape) == 1:

        nx = x.shape[0]
        nz = z.shape[0]

        for xi in _utils.down_sample(x, N):
            ax.plot(xi.repeat(nz), z, 'k', linewidth=0.5)

        for zi in _utils.down_sample(z, N):
            ax.plot(x, zi.repeat(nx), 'k', linewidth=0.5)

    elif len(x.shape) == 2 and len(z.shape) == 2:

        nx = x.shape[0]
        nz = x.shape[1]

        for i in _np.arange(0, nz, int(nx/N)):
            ax.plot(x[:, i], z[:, i], 'k')

        for i in _np.arange(0, nx, int(nz/N)):
            ax.plot(x[i, :], z[i, :], 'k')

        ax.plot(x[:, -1], z[:, -1], 'k')
        ax.plot(x[-1, :], z[-1, :], 'k')

    else:
        raise ValueError('x and z must be 1 or 2d arrays')


def patch_text(ax, patch, text, rotation=0, color='k'):
    """ Add text to matplotlib patch.

    Parameters
    ----------

    ax : matplotlib axe
    patch : matplotlib patch
    text : string to display in patch
    """
    rx, ry = patch.get_xy()
    cx = rx + patch.get_width()/2.0
    cy = ry + patch.get_height()/2.0
    ax.annotate(text, (cx, cy), color=color, weight='bold',
                fontsize=12, ha='center', va='center', rotation=rotation)


def bc_text(ax, x, z, dx, dz, bc, offset=10, bg='w', fg='k'):
    """ Display bc conditions. """

    loc = [(x[0]-offset*dx, z[0]-offset*dz), (x[0], z[0]-offset*dz),
           (x[-1], z[0]-offset*dz), (x[0], z[-1])]
    width = [offset*dx, x[-1] - x[0]]*2
    height = [z[-1] - z[0] + 2*offset*dz, offset*dz]*2

    for i, l, w, h in zip(range(4), loc, width, height):

        rect = _patches.Rectangle(l, w, h, facecolor=bg)
        patch_text(ax, rect, bc[i], color=fg)
        ax.add_patch(rect)


def plot_subdomains(ax, x, z, domain, legend=False,
                    facecolor='k', edgecolor='k', alpha=0.5, curvilinear=False):
    """ Plot subdomain in ax.

    Obstacle can be a list of coordinate lists or a Domain object.

    Parameters
    ----------

    x, z : 1D arrays. x and z coordinates
    ax : Matplotlib axe where the subdomains will be plotted.
    subdomains : List of coordinates or Domain object.
    facecolor : Fill color. Optional.
    edgecolor ; Line color. Optional.
    curvilinear : Boolean. Optional.
    """

    for sub in domain:

        if curvilinear and isinstance(sub, Subdomain):
            for i in range(2):
                ax.plot(x[sub.rx, sub.iz[i]], z[sub.rx, sub.iz[i]],
                        color=edgecolor, linewidth=3)
                ax.plot(x[sub.ix[i], sub.rz], z[sub.ix[i], sub.rz],
                        color=edgecolor, linewidth=3)

            ax.fill_between(x[sub.rx, sub.iz[0]],
                            z[sub.rx, sub.iz[0]],
                            z[sub.rx, sub.iz[1]], color=facecolor, alpha=alpha)

        elif curvilinear and isinstance(sub, (_np.ndarray, list)):
            ax.plot(x[sub[0], sub[1]:sub[3]+1], z[sub[0], sub[1]:sub[3]+1],
                    color=edgecolor, linewidth=3)
            ax.plot(x[sub[2], sub[1]:sub[3]+1], z[sub[2], sub[1]:sub[3]+1],
                    color=edgecolor, linewidth=3)

            ax.plot(x[sub[0]:sub[2]+1, sub[1]], z[sub[0]:sub[2]+1, sub[1]],
                    color=edgecolor, linewidth=3)
            ax.plot(x[sub[0]:sub[2]+1, sub[3]], z[sub[0]:sub[2]+1, sub[3]],
                    color=edgecolor, linewidth=3)

        elif isinstance(sub, (_np.ndarray, list, Subdomain)):
            if isinstance(sub, (_np.ndarray, list)):
                origin = (x[sub[0]], z[sub[1]])
                width = x[sub[2]] - x[sub[0]]
                height = z[sub[3]] - z[sub[1]]
            elif isinstance(sub, Subdomain):
                origin = (x[sub.ix[0]], z[sub.iz[0]])
                width = x[sub.ix[1]] - x[sub.ix[0]]
                height = z[sub.iz[1]] - z[sub.iz[0]]

            rect = _patches.Rectangle(origin, width, height, linewidth=3,
                                      edgecolor=edgecolor, facecolor=facecolor,
                                      alpha=alpha)
            ax.add_patch(rect)

        else:
            msg = 'Each element of subtacle must be a list, array, or Subdomain object'
            raise ValueError(msg)

        if legend and isinstance(sub, Subdomain):
            if sub.tag in ['X', 'A', 'W']:
                patch_text(ax, rect, sub.key, color=edgecolor)


def plot_pml(ax, x, z, bc, Npml, ecolor='k', fcolor='k'):
    """ Display PML areas. """

    alpha = 0.1

    if bc[0] == 'A':
        rect = _patches.Rectangle((x[0], z[0]),
                                  x[Npml] - x[0],
                                  z[-1] - z[0],
                                  linewidth=3,
                                  edgecolor=ecolor, facecolor=fcolor,
                                  alpha=alpha)
        patch_text(ax, rect, 'PML', rotation=90)
        ax.add_patch(rect)

    if bc[1] == 'A':
        rect = _patches.Rectangle((x[0], z[0]),
                                  z[Npml] - z[0],
                                  x[-1] - x[0],
                                  linewidth=3,
                                  edgecolor=ecolor, facecolor=fcolor,
                                  alpha=alpha)
        patch_text(ax, rect, 'PML')
        ax.add_patch(rect)

    if bc[2] == 'A':
        rect = _patches.Rectangle((x[-Npml], z[0]),
                                  x[-1] - x[-Npml],
                                  z[-1] - z[0],
                                  linewidth=3,
                                  edgecolor=ecolor, facecolor=fcolor,
                                  alpha=alpha)
        patch_text(ax, rect, 'PML', rotation=90)
        ax.add_patch(rect)

    if bc[3] == 'A':
        rect = _patches.Rectangle((x[0], z[0]),
                                  z[-1] - z[-Npml],
                                  x[-1] - x[0],
                                  linewidth=3,
                                  edgecolor=ecolor, facecolor=fcolor,
                                  alpha=alpha)
        patch_text(ax, rect, 'PML')
        ax.add_patch(rect)
