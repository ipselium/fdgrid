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

-----------
"""

import numpy as _np
from matplotlib import patches as _patches, path as _path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mplutils import modified_jet
from fdgrid import utils as _utils, Subdomain, Domain, Obstacle


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
    patch : matplotlib patch (Only Rectangle and PathPatch for now)
    text : string to display in patch
    """

    if isinstance(patch, _patches.Rectangle):
        rx, ry = patch.get_xy()
        cx = rx + patch.get_width()/2.0
        cy = ry + patch.get_height()/2.0

    elif isinstance(patch, _patches.PathPatch):
        c = patch.get_path().get_extents().get_points()
        cx, cy = (c[0][0] + c[1][0])/2, (c[0][1] + c[1][1])/2

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


def _check_domain(x, z, domain):

    if isinstance(domain, (_np.ndarray, list, tuple)):
        if isinstance(domain, _np.ndarray):
            domain = domain.tolist()
        domain = Domain((x.shape[0], z.shape[0]), data=domain)

    if not isinstance(domain, Domain):
        raise ValueError('domain must be a Domain, list, tuple or array object')

    return domain


def plot_subdomains(ax, x, z, domain, legend=False, facecolor='k', edgecolor='k', alpha=0.5):
    """ Plot subdomain in ax.

    Parameters
    ----------

    x, z : 1D arrays. x and z coordinates
    ax : Matplotlib axe where the subdomains will be plotted.
    domain : List/Tuple of coordinates or Domain object.
    facecolor : Fill color. Optional.
    edgecolor ; Line color. Optional.
    curvilinear : Boolean. Optional.
    """

    domain = _check_domain(x, z, domain)

    if len(x.shape) > 1 and len(z.shape) > 1:
        dim = 2
        # Edges
        ax.plot(x[0, :], z[0, :], 'k', linewidth=3)
        ax.plot(x[-1, :], z[-1, :], 'k', linewidth=3)
        ax.plot(x[:, 0], z[:, 0], 'k', linewidth=3)
        ax.plot(x[:, -1], z[:, -1], 'k', linewidth=3)
    else:
        dim = 1
        edges = _patches.Rectangle((x[0], z[0]), x[-1]-x[0], z[-1]-z[0],
                                   linewidth=3, fill=None)
        ax.add_patch(edges)

    for sub in domain:

        if dim == 1:
            origin = (x[sub.ix[0]], z[sub.iz[0]])
            width = x[sub.ix[1]] - x[sub.ix[0]]
            height = z[sub.iz[1]] - z[sub.iz[0]]

            patch = _patches.Rectangle(origin, width, height, linewidth=3,
                                       edgecolor=edgecolor, facecolor=facecolor,
                                       alpha=alpha)
            ax.add_patch(patch)

        elif dim == 2:

            a = [(x[i, sub.iz[0]], z[i, sub.iz[0]]) for i in sub.rx][::-1]
            b = [(x[sub.ix[0], i], z[sub.ix[0], i]) for i in sub.rz]
            c = [(x[i, sub.iz[1]], z[i, sub.iz[1]]) for i in sub.rx]
            d = [(x[sub.ix[1], i], z[sub.ix[1], i]) for i in sub.rz]

            verts = a + b + c + d
            codes = [_path.Path.MOVETO] + \
                    (len(verts)-2)*[_path.Path.LINETO] + \
                    [_path.Path.CLOSEPOLY]
            path = _path.Path(verts, codes)
            patch = _patches.PathPatch(path, linewidth=3,
                                       edgecolor=edgecolor, facecolor=facecolor,
                                       alpha=alpha)
            ax.add_patch(patch)

        if legend and isinstance(sub, Obstacle):
            patch_text(ax, patch, sub.key, color=edgecolor)

        elif legend and isinstance(sub, Subdomain):
            if sub.tag in ['X', 'A', 'W']:
                patch_text(ax, patch, sub.key, color=edgecolor)


def plot_bc_profiles(ax, x, z, obstacles, color='r'):
    """ Plot obstacle boundary profiles in ax.

    Parameters
    ----------

    x, z : 1D arrays. x and z coordinates
    ax : Matplotlib axe where the subdomains will be plotted.
    obstacles : Domain object.
    """

    cm = modified_jet()

    for i, obs in enumerate(obstacles):
        xsize = 0.02*(x.max()-x.min())
        zsize = 0.02*(z.max()-z.min())

        for bc in obs.edges:

            if bc.axis == 1 and len(x.shape) == 1:
                ax.plot(x[bc.sx], z[bc.sz] + xsize*bc.vn/abs(bc.vn).max(),
                        color=color, linewidth=3, label=f'obs : {i+1}')

            elif bc.axis == 0 and len(x.shape) == 1:
                ax.plot(x[bc.sx] + zsize*bc.vn/abs(bc.vn).max(), z[bc.sz],
                        color=color, linewidth=3, label=f'obs : {i+1}')

            elif bc.axis == 1 and len(x.shape) > 1:
                sz = slice(bc.sz-1, bc.sz+2)
                im = ax.pcolormesh(x[bc.sx, sz], z[bc.sx, sz],
                                   _np.tile(bc.vn, (sz.stop-sz.start, 1)).T,
                                   cmap=cm, alpha=1)

            elif bc.axis == 0 and len(x.shape) > 1:
                sx = slice(bc.sx-1, bc.sx+2)
                im = ax.pcolormesh(x[sx, bc.sz], z[sx, bc.sz],
                                   _np.tile(bc.vn, (sz.stop-sz.start, 1)).T,
                                   cmap=cm)

    if 'im' in locals():
        cbaxes = inset_axes(ax, width="30%", height="3%", loc=3)
        ax.get_figure().colorbar(im, cax=cbaxes, orientation='horizontal')
        cbaxes.xaxis.set_ticks_position('top')


def plot_pml(ax, x, z, bc, Npml, ecolor='k', fcolor='k', alpha=0.1):
    """ Display PML areas. """

    if len(x.shape) > 1 and len(z.shape) > 1:
        _plot_pml_dim2(ax, x, z, bc, Npml, ecolor, fcolor, alpha)
    else:
        _plot_pml_dim1(ax, x, z, bc, Npml, ecolor, fcolor, alpha)


def _plot_pml_dim1(ax, x, z, bc, Npml, ecolor, fcolor, alpha):
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
                                  x[-1] - x[0],
                                  z[Npml] - z[0],
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
        rect = _patches.Rectangle((x[0], z[-Npml]),
                                  x[-1] - x[0],
                                  z[-1] - z[-Npml],
                                  linewidth=3,
                                  edgecolor=ecolor, facecolor=fcolor,
                                  alpha=alpha)
        patch_text(ax, rect, 'PML')
        ax.add_patch(rect)


def _plot_pml_dim2(ax, x, z, bc, Npml, edgecolor, facecolor, alpha):

    if bc[0] == 'A':
        a = [(x[i, 0], z[i, 0]) for i in range(Npml)]
        b = [(x[Npml, i], z[Npml, i]) for i in range(z.shape[1])]
        c = [(x[i, -1], z[i, -1]) for i in range(Npml, -1, -1)]
        d = [(x[0, i], z[0, i]) for i in range(z.shape[1]-1, -1, -1)]

        verts = a + b + c + d
        codes = [_path.Path.MOVETO] + \
                (len(verts)-2)*[_path.Path.LINETO] + \
                [_path.Path.CLOSEPOLY]
        path = _path.Path(verts, codes)
        patch = _patches.PathPatch(path, linewidth=3,
                                   edgecolor=edgecolor, facecolor=facecolor,
                                   alpha=alpha)

        patch_text(ax, patch, 'PML', rotation=90)
        ax.add_patch(patch)

    if bc[1] == 'A':
        a = [(x[i, 0], z[i, 0]) for i in range(x.shape[0])]
        b = [(x[-1, i], z[-1, i]) for i in range(Npml)]
        c = [(x[i, Npml], z[i, Npml]) for i in range(x.shape[0]-1, -1, -1)]
        d = [(x[0, i], z[0, i]) for i in range(Npml-1, -1, -1)]
        verts = a + b + c + d
        codes = [_path.Path.MOVETO] + \
                (len(verts)-2)*[_path.Path.LINETO] + \
                [_path.Path.CLOSEPOLY]
        path = _path.Path(verts, codes)
        patch = _patches.PathPatch(path, linewidth=3,
                                   edgecolor=edgecolor, facecolor=facecolor,
                                   alpha=alpha)

        patch_text(ax, patch, 'PML')
        ax.add_patch(patch)

    if bc[2] == 'A':
        a = [(x[i, 0], z[i, 0]) for i in range(x.shape[0] - Npml, x.shape[0])]
        b = [(x[x.shape[0] - Npml, i], z[x.shape[0] - Npml, i]) for i in range(z.shape[1])]
        c = [(x[i, -1], z[i, -1]) for i in range(x.shape[0]-1, x.shape[0] - Npml, -1)]
        d = [(x[-1, i], z[-1, i]) for i in range(z.shape[1]-1, -1, -1)]
        verts = a + b + c + d
        codes = [_path.Path.MOVETO] + \
                (len(verts)-2)*[_path.Path.LINETO] + \
                [_path.Path.CLOSEPOLY]
        path = _path.Path(verts, codes)
        patch = _patches.PathPatch(path, linewidth=3,
                                   edgecolor=edgecolor, facecolor=facecolor,
                                   alpha=alpha)

        patch_text(ax, patch, 'PML', rotation=90)
        ax.add_patch(patch)

    if bc[3] == 'A':
        a = [(x[i, z.shape[1]-Npml], z[i, z.shape[1]-Npml]) for i in range(x.shape[0])]
        b = [(x[-1, i], z[-1, i]) for i in range(z.shape[1] - Npml, z.shape[1])]
        c = [(x[i, -1], z[i, -1]) for i in range(x.shape[0]-1, -1, -1)]
        d = [(x[0, i], z[0, i]) for i in range(z.shape[1]-1, z.shape[1]-Npml, -1)]
        verts = a + b + c + d
        codes = [_path.Path.MOVETO] + \
                (len(verts)-2)*[_path.Path.LINETO] + \
                [_path.Path.CLOSEPOLY]
        path = _path.Path(verts, codes)
        patch = _patches.PathPatch(path, linewidth=3,
                                   edgecolor=edgecolor, facecolor=facecolor,
                                   alpha=alpha)

        patch_text(ax, patch, 'PML')
        ax.add_patch(patch)
