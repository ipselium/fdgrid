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
# Creation Date : 2019-02-13 - 10:30:40
#
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
"""
-----------

Module `mesh` provides three classes to build meshes:

    * :py:class:`Mesh`: Build regular cartesian mesh
    * :py:class:`AdaptativeMesh`: Build adaptative cartesian mesh
    * :py:class:`CurvilinearMesh`: Build curvilinear mesh

-----------
"""

__all__ = ['Mesh', 'AdaptativeMesh', 'CurvilinearMesh']

import re as _re
import itertools as _itertools
import numpy as _np
import matplotlib.pyplot as _plt
import matplotlib.ticker as _ticker
import ofdlib2.derivation as _drv
import mpl_toolkits.axes_grid1 as _axes_grid1
from fdgrid import _exceptions
from fdgrid import graphics as _graphics
from fdgrid.cdomain import ComputationDomains
from fdgrid.templates import curv as _curv


class Mesh:
    """ Build cartesian regular grid

    Parameters
    ----------
    shape : tuple
        Size of the domain. Must be a tuple with two int objects.
    step : tuple
        Spatial steps. Must be a tuple with two float objects.
    origin : tuple, optional
        Origin of the grid. Must be a tuple with two int objects.
    bc : {'[ARZPW][ARZPW][ARZPW][ARZPW]'}, optional
        Boundary conditions. Must be a 4 characters string.
    obstacles : :py:class:`fdgrid.domains.Domain`, optional
        Obstacles in the computation domain.
    Npml : int, optional
        Number of points of the absorbing area (only if 'A' in `bc`).
    stencil : int, optional
        Size of the finite difference stencil (used by :py:mod:`nsfds2`).

    See also
    --------
    :py:class:`AdaptativeMesh`,
    :py:class:`CurvilinearMesh`,
    :py:mod:`fdgrid.templates`

    """

    def __init__(self, shape, step, origin=(0, 0),
                 bc='WWWW', obstacles=None, Npml=15, stencil=11):


        self.shape = shape
        self.nx, self.nz = shape
        self.dx, self.dz = step
        self.ix0, self.iz0 = origin
        self.stencil = stencil
        self.Npml = Npml

        self.obstacles = obstacles
        if not self.obstacles:
            self.obstacles = []
        self.bc = bc.upper()
        self._check_bc()
        self._check_grid()

        self._make_grid()
        self._find_subdomains()
        self._make_moving_bc()

    def _make_grid(self):
        """ Make grid. """

        self.x = (_np.arange(self.nx)-self.ix0)*self.dx
        self.z = (_np.arange(self.nz)-self.iz0)*self.dz

    def N(self, axis):
        """ Returns the number of points following `axis`. """
        return self.nx if axis == 0 else self.nz if axis == 1 else None

    def du(self, axis):
        """ Returns the spacial base step following `axis`. """
        return self.dx if axis == 0 else self.dz if axis == 1 else None

    def _limits(self):
        xlim = ComputationDomains.bounds(self.shape, self.bc, self.obstacles, axis=0)
        zlim = ComputationDomains.bounds(self.shape, self.bc, self.obstacles, axis=1)
        return xlim, zlim

    def _find_subdomains(self):
        """ Divide the computation domain into subdomains.

        dx/dzdomains : for x/z derivatives
        fx/fzdomains : for x/z filters

        dsdomains : smallest domain among dx/dzdomains
        fsdomains : smallest domain among fx/fzdomains
        adomains : for PML

        """

        self.domain = ComputationDomains((self.nx, self.nz), self.obstacles,
                                         self.bc, self.stencil, self.Npml)

        self.dxdomains = self.domain.dxdomains
        self.dzdomains = self.domain.dzdomains
        self.dsdomains = self.domain.dsdomains
        self.dmdomains = self.dxdomains + self.dzdomains

        self.fxdomains = self.domain.fxdomains
        self.fzdomains = self.domain.fzdomains
        self.fsdomains = self.domain.fsdomains
        self.fmdomains = self.fxdomains + self.fzdomains

        self.adomains = self.domain.adomains
        self.gdomain = self.domain.gdomain

    def _check_bc(self):

        regex = [r'[^P].P.', r'P.[^P].', r'.[^P].P', r'.P.[^P]']

        if not _re.match(r'^[ZRAPW]*$', self.bc):
            raise _exceptions.BoundaryConditionError("bc must be combination of 'ZRAPW'!")

        if any(_re.match(r, self.bc) for r in regex):
            msg = "periodic condition must be on both sides of the domain,"
            msg += " i.e. '(P.P.)'|'(.P.P)'"
            raise _exceptions.BoundaryConditionError(msg)

    def _check_grid(self):

        if self.Npml < self.stencil:
            raise _exceptions.GridError("Number of points of PML must be larger than stencil")

        if self.ix0 > self.nx or self.iz0 > self.nz:
            raise _exceptions.GridError("Origin of the domain must be in the domain")

    def _make_moving_bc(self):

        for obs in self.obstacles:
            obs.make_moving_bc(self.x, self.z)

    def plot_domains(self, legend=False, N=4, figsize=(9, 18),
                     bc_profiles=False, filename=None):
        """ Plot a scheme of the computation domain higlighting the subdomains.

        Parameters
        ----------
        legend: bool, optional
            Show legend.
        N: int, optional
            Plot the grid with one point over N.
        filename: string, optional
            Save to `filename`.
        figsize: tuple, optional
            Size of the figure.
        bc_profiles: bool, optional
            Show velocity profiles for moving boundaries

        See also
        --------
        :py:meth:`plot_grid`,
        :py:meth:`plot_physical`,
        :py:meth:`plot_xz`

        """

        _, axes = _plt.subplots(2, 1, figsize=figsize)

        # Grid & Obstacles
        for ax in axes:
            ax.set_xlabel('x [m]')
            ax.set_ylabel('z [m]')
            ax.set_aspect('equal')

            _graphics.plot_grid(ax, self.x, self.z, N=N)
            _graphics.plot_subdomains(ax, self.x, self.z, self.obstacles)

        # Subdomains
        for tag, color in zip(['X', 'W', 'A'], ['b', 'r', 'g']):
            _graphics.plot_subdomains(axes[0], self.x, self.z,
                                      [s for s in self.dxdomains if s.tag == tag],
                                      facecolor='y', edgecolor=color, legend=legend)
            _graphics.plot_subdomains(axes[1], self.x, self.z,
                                      [s for s in self.dzdomains if s.tag == tag],
                                      facecolor='y', edgecolor=color, legend=legend)
            _graphics.plot_subdomains(axes[0], self.x, self.z,
                                      [s for s in self.adomains if s.tag == tag],
                                      facecolor='y', edgecolor=color, legend=legend)
            _graphics.plot_subdomains(axes[1], self.x, self.z,
                                      [s for s in self.adomains if s.tag == tag],
                                      facecolor='y', edgecolor=color, legend=legend)

        if legend:
            for ax in axes:
                _graphics.bc_text(ax, self.x, self.z, self.dx, self.dz, self.bc,
                                  bg='k', fg='w')
            print(self.domain)

        if bc_profiles:
            for ax in axes:
                _graphics.plot_bc_profiles(ax, self.x, self.z, self.obstacles)


        _plt.tight_layout()

        if isinstance(filename, str):
            _plt.savefig(filename)

    def plot_grid(self, figsize=(9, 5), N=4, axis=False, legend=False,
                  pml=False, bc_profiles=False, probes=False, filename=None):
        """ Numerical grid representation.

        Parameters
        ----------

        legend: bool, optional
            Show legend.
        N: int, optional
            Plot the grid with one point over N.
        filename: string, optional
            Save to `filename`.
        figsize: tuple, optional
            Size of the figure.
        pml: bool, optional
            Show absorbing area in the figure.
        bc_profiles: bool, optional
            Show velocity profiles for moving boundaries

        Important
        ---------
        The curves :math:`dx'/dx` are not representative on the edges because
        of the gradient.

        See also
        --------
        :py:meth:`plot_physical`,
        :py:meth:`plot_xz`

        """

        fig, ax_c = _plt.subplots(figsize=figsize)
        fig.subplots_adjust(.1, .1, .95, .95)

        _graphics.plot_subdomains(ax_c, self.x, self.z, self.obstacles,
                                  legend=legend, facecolor='k')
        _graphics.plot_grid(ax_c, self.x, self.z, N)

        ax_c.set_xlim(self.x.min(), self.x.max())
        ax_c.set_ylim(self.z.min(), self.z.max())
        ax_c.set_xlabel(r'$x$ [m]')
        ax_c.set_ylabel(r'$z$ [m]')
        ax_c.set_aspect('equal')

        if probes:
            _ = [ax_c.plot(self.x[i], self.z[j], 'ro') for i, j in probes]

        if pml:
            _graphics.plot_pml(ax_c, self.x, self.z, self.bc, self.Npml)

        if bc_profiles:
            _graphics.plot_bc_profiles(ax_c, self.x, self.z, self.obstacles)

        if axis:
            divider = _axes_grid1.make_axes_locatable(ax_c)
            ax_xa = divider.append_axes('top', size='30%', pad=0.1)
            ax_xb = divider.append_axes('top', size='20%', pad=0.1)
            ax_za = divider.append_axes('right', size='15%', pad=0.1)
            ax_zb = divider.append_axes('right', size='10%', pad=0.1)

            ax_xa.plot(self.x, _np.gradient(self.x)/self.dx, 'ko')
            ax_xa.set_ylabel(r"$x'/dx$")
            ax_xb.plot(self.x, range(len(self.x)), 'k', linewidth=2)
            ax_xb.set_ylabel(r"$N_x$")

            ax_za.plot(_np.gradient(self.z)/self.dz, self.z, 'ko')
            ax_za.set_xlabel(r"$z'/dz$")
            ax_zb.plot(range(len(self.z)), self.z, 'k', linewidth=2)
            ax_zb.set_xlabel(r"$N_z$")

            for ax in [ax_xa, ax_xb]:
                ax.set_xlim(ax_c.get_xlim())
                ax.xaxis.set_major_formatter(_ticker.NullFormatter())  # no label
                for j in self._limits()[0]:
                    if j[-1] == 'o':
                        ax.axvspan(self.x[j[0]], self.x[j[1]], facecolor='k', alpha=0.5)

            for ax in [ax_za, ax_zb]:
                ax.set_ylim(ax_c.get_ylim())
                ax.yaxis.set_major_formatter(_ticker.NullFormatter())  # no label
                for j in self._limits()[1]:
                    if j[-1] == 'o':
                        ax.axhspan(self.z[j[0]], self.z[j[1]], facecolor='k', alpha=0.5)

        if isinstance(filename, str):
            _plt.savefig(filename)

    def plot_physical(self, figsize=(9, 4), legend=False,
                      pml=False, bc_profiles=False, probes=False, filename=None):
        """ Physical grid representation.


        See also
        --------
        :py:meth:`plot_grid`,
        :py:meth:`plot_xz`

        """

        self.plot_grid(figsize=figsize, pml=pml, legend=legend,
                       bc_profiles=bc_profiles, probes=probes, filename=filename)

    def plot_xz(self, figsize=(9, 4), filename=None):
        """ Plot x & z axis.

        Parameters
        ----------
        filename : str, optional
            Save figure to filename.
        figsize : tuple, optional
            Size of the figure.

        See also
        --------
        :py:meth:`plot_physical`,
        :py:meth:`plot_xz`

        """

        _, axes = _plt.subplots(2, 2, figsize=figsize)
        axes[0, 0].plot(self.x, 'k*')
        axes[0, 0].set_xlabel(r'$N_x$')
        axes[0, 0].set_ylabel(r'$x$ [m]')

        axes[1, 0].plot(_np.gradient(self.x)/self.dx, 'k*')
        axes[1, 0].set_xlabel(r'$x$ [m]')
        axes[1, 0].set_ylabel(r"$x'/dx$")

        axes[0, 1].plot(self.z, 'k*')
        axes[0, 1].set_xlabel(r'$N_z$')
        axes[0, 1].set_ylabel(r'$z$ [m]')

        axes[1, 1].plot(_np.gradient(self.z)/self.dz, 'k*')
        axes[1, 1].set_xlabel(r'$z$ [m]')
        axes[1, 1].set_ylabel(r"$z'/dz$")

        for i in range(axes.shape[1]):
            for ax in axes[:, i]:
                ax.grid()
                for j in self._limits()[i]:
                    if j[-1] == 'o':
                        ax.axvspan(j[0], j[1], facecolor='k', alpha=0.5)

        _plt.tight_layout()

        if isinstance(filename, str):
            _plt.savefig(filename)

    def get_obstacles(self):
        """ Get a list of the coordinates of all obstacles. """
        return [sub.xz for sub in self.obstacles]

    @staticmethod
    def show_figures():
        """ Show all figures. """
        _plt.show()

    def __str__(self):
        s = 'Cartesian {}x{} points grid with {} boundary conditions:\n\n'
        s += '\t* Spatial step : {}\n'.format((self.dx, self.dz))
        s += '\t* Origin       : {}\n'.format((self.ix0, self.iz0))
        if 'A' in self.bc:
            s += '\t* Points in PML: {}\n'.format(self.Npml)
        s += '\t* Max stencil  : {}\n'.format(self.stencil)

        return s.format(self.nx, self.nz, self.bc)

    def __repr__(self):
        return self.__str__()


class AdaptativeMesh(Mesh):
    """ Build cartesian adaptative grid

    Parameters
    ----------
    dilatation : float
        dilatation rate over `Nd` points.
    Nd : int
        Number of point of adaptative areas.
    only_pml : bool
        Only adapt grid in PML areas.

    See also
    --------
    :py:class:`Mesh`,
    :py:mod:`fdgrid.templates`

    """

    def __init__(self, shape, step, origin=(0, 0),
                 bc='WWWW', obstacles=None, Npml=15, stencil=11,
                 dilatation=3., Nd=23, only_pml=False):

        self.only_pml = only_pml
        self.N = Nd
        self.dilatation = dilatation/100

        super().__init__(shape, step, origin, bc, obstacles, Npml, stencil)

    def _make_grid(self):

        self.stretch = 1 - _np.arange(self.N)/self.N*self.dilatation
        self.stretch_PML = 1 + ((_np.arange(self.Npml) - self.Npml)/self.Npml)**2
        self.amin = self.stretch[-1]

        xlim, zlim = self._limits()

        self.ax = self._make_axis(self.nx, xlim, axis=0)
        self.az = self._make_axis(self.nz, zlim, axis=1)

        self.x = _np.cumsum(self.ax*self.dx)
        self.x -= self.x[self.ix0]

        self.z = _np.cumsum(self.az*self.dz)
        self.z -= self.z[self.iz0]

    def _make_axis(self, Nm, limits, axis):

        # Stretch coeff
        a = _np.zeros(Nm)

        if len(limits) == 1:
            self._grid_single(a, limits, axis)
        else:
            self._grid_multi(a, limits, axis)

        return a

    def _grid_single(self, a, limits, axis):

        start, stop, _ = limits[0]
        a[:] = 1

        if self.bc[axis] == 'A':
            a[:self.Npml] = self.stretch_PML

        elif self.bc[axis] in ['W', 'Z']:
            a[start:start+self.stencil] = self.amin
            a[start+self.stencil:start+self.stencil+self.N] = self.stretch[::-1]

        if self.bc[axis+2] == 'A':
            a[stop-self.Npml+1:] = self.stretch_PML[::-1]

        elif self.bc[axis+2] in ['W', 'Z']:
            a[stop-self.stencil:stop+1] = self.amin
            a[stop-self.stencil-self.N+1:stop-self.stencil+1] = self.stretch

    def _grid_multi(self, a, limits, axis):

        mkrs = self._markers(limits, axis)

        for [start, stop, _], mk in zip(limits, mkrs):

            if mk == [2, 1]:
                a[:self.Npml] = self.stretch_PML
                a[self.Npml:stop+1] = 1

            elif mk == [2, self.amin]:
                a[:self.Npml] = self.stretch_PML
                a[self.Npml:stop-self.stencil-self.N] = 1
                a[stop-self.stencil-self.N:stop-self.stencil] = self.stretch[:self.N]
                a[stop-self.stencil:stop+1] = self.amin

            elif mk == [1, 2]:
                a[stop-self.Npml+1:] = self.stretch_PML[::-1]
                a[start:stop-self.Npml+1] = 1

            elif mk == [self.amin, 2]:
                a[stop-self.Npml+1:] = self.stretch_PML[::-1]
                a[start:start+self.stencil] = self.amin
                a[start+self.stencil:start+self.stencil+self.N] = self.stretch[:self.N][::-1]
                a[start+self.stencil+self.N:stop-self.Npml+1] = 1

            else:
                self._middle(a, start, stop, mk)

    def _middle(self, a, start, stop, mk):

        # Left
        a[start:min(stop+1, start+self.stencil)] = mk[0]

        # Right
        a[max(start, stop-self.stencil):stop+1] = mk[1]

        # Center
        if  stop - start - 2*self.stencil > 2:

            Np = min(self.N, int((stop - start - 2*self.stencil)/2))

            # Increase
            if mk[0] == self.amin:
                a[start+self.stencil:start+self.stencil+Np] = self.stretch[::-1][:Np]
            elif mk[0] == 1:
                a[start+self.stencil:start+self.stencil+Np] = 1

            # Decrease
            if mk[1] == self.amin:
                a[stop-self.stencil-Np+1:stop-self.stencil+1] = self.stretch[self.N-Np:]
            elif mk[1] == 1:
                a[stop-self.stencil-Np+1:stop-self.stencil+1] = 1

            # Plateau
            a[start+self.stencil+Np:stop-self.stencil-Np+1] = a[start+self.stencil+Np-1]

        else:
            a[start+self.stencil:stop+1] = a[start]

    def _markers(self, limits, axis):

        mkrs = []
        self._markers_left(mkrs, limits, axis)
        self._markers_center(mkrs, limits)
        self._markers_right(mkrs, limits, axis)

        return self._make_compatible(mkrs)

    def _markers_left(self, mkrs, limits, axis):

        start, stop, _ = limits[0]
        if self.bc[axis] == 'A':
            if stop - start >= self.Npml + self.N + self.stencil:
                mkrs.append((2, self.amin))
            else:
                mkrs.append((2, 1))

        elif self.bc[axis] == 'P':
            if stop - start >= 2*self.stencil + self.N:
                mkrs.append((1, self.amin))
            else:
                mkrs.append((1, 1))

        else:
            mkrs.append((self.amin, self.amin))

    def _markers_center(self, mkrs, limits):
        """ WARNING : The case of one variation is not processed.
        Treated as is no variation is possible! """

        for start, stop, _ in limits[1:-1]:
            if stop - start < 2*self.stencil + self.N:
                mkrs.append((1, 1))

            elif 2*self.stencil + self.N <= stop - start < 2*self.stencil + 2*self.N:
                mkrs.append((1, 1)) # Processes as if no variation is possible

            elif stop - start >= 2*self.stencil + 2*self.N:
                mkrs.append((self.amin, self.amin))

    def _markers_right(self, mkrs, limits, axis):

        start, stop, _ = limits[-1]
        if self.bc[axis+2] == 'A':
            if stop - start >= self.Npml + self.N + self.stencil:
                mkrs.append((self.amin, 2))
            else:
                mkrs.append((1, 2))

        elif self.bc[axis+2] == 'P':
            if stop - start >= 2*self.stencil + self.N:
                mkrs.append((self.amin, 1))
            else:
                mkrs.append((1, 1))

        else:
            mkrs.append((self.amin, self.amin))

    def _make_compatible(self, mkrs, LOG=False):

        d = {(2, 1): [(2, 1)],
             (1, 2): [(1, 2)],
             (2, self.amin): [(2, 1), (2, self.amin)],
             (self.amin, 2): [(self.amin, 2), (1, 2)],
             (self.amin, self.amin): [(self.amin, self.amin),
                                      (1, 1),
                                      (self.amin, 1),
                                      (1, self.amin)],
             (1, 1): [(1, 1), (self.amin, self.amin)],
             (1, self.amin): [(1, self.amin), (1, 1)],
             (self.amin, 1): [(self.amin, 1), (1, 1)]}

        alt = list(_itertools.product(*(d[k] for k in mkrs)))
        alt = [k.tolist() for k in _np.array(alt) if _np.all(k[:-1, 1] == k[1:, 0])]
        if len(alt) > 1:
            if LOG:
                print('{} alternative meshes'.format(len(alt)))
            alt = alt[_np.argmin(_np.array([k.mean() for k in _np.array(alt)]))]
        else:
            alt = alt[0]

        if LOG:
            print(mkrs)
            print(alt)

        return alt

    def __str__(self):
        s = 'Adaptative cartesian {}x{} points grid with {} boundary conditions:\n\n'
        s += '\t* Spatial step  : {}\n'.format((self.dx, self.dz))
        s += '\t* Origin        : {}\n'.format((self.ix0, self.iz0))
        if 'A' in self.bc:
            s += '\t* Points in PML : {}\n'.format(self.Npml)
        s += '\t* Max stencil   : {}\n'.format(self.stencil)
        s += '\t* Dilatation over {} pts  : {:.1f} %\n'.format(self.N, (1-self.amin)*100)

        return s.format(self.nx, self.nz, self.bc)


class CurvilinearMesh(Mesh):
    """ Build curvilinear grid

    Parameters
    ----------
    fcurvxz : function, optional
        Function taking as input arguments the numerical coordinates ('xn', 'yn')
        and returning the physical coordinates ('xp', 'zp').

    See also
    --------
    :py:class:`Mesh`,
    :py:mod:`fdgrid.templates`
    """

    def __init__(self, shape, step, origin=(0, 0),
                 bc='WWWW', obstacles=None, Npml=15, stencil=11, fcurvxz=_curv):

        if not fcurvxz:
            self.fcurvxz = _curv
        else:
            self.fcurvxz = fcurvxz

        super().__init__(shape, step, origin, bc, obstacles, Npml, stencil)

    def plot_physical(self, figsize=(9, 4), legend=False,
                      pml=False, bc_profiles=False, probes=False, filename=None):
        """ Plot physical and numerical domains.

        Parameters
        ----------
        legend: bool, optional
            Show legend.
        N: int, optional
            Plot the grid with one point over N.
        filename: string, optional
            Save to `filename`.
        figsize: tuple, optional
            Size of the figure.
        pml: bool, optional
            Show absorbing area in the figure.
        bc_profiles: bool, optional
            Show velocity profiles for moving boundaries
        """

        edgecolor = 'k'
        facecolor = 'k'
        alpha = 0.5

        _, axes = _plt.subplots(ncols=2, figsize=figsize)
        _graphics.plot_grid(axes[0], self.xn, self.zn)
        _graphics.plot_grid(axes[1], self.xp, self.zp)
        axes[0].set(title='Numerical Domain')
        axes[1].set(title='Physical Domain')

        _graphics.plot_subdomains(axes[0], self.x, self.z, self.obstacles,
                                  legend=legend,
                                  edgecolor=edgecolor, facecolor=facecolor, alpha=alpha)
        _graphics.plot_subdomains(axes[1], self.xp, self.zp, self.obstacles,
                                  legend=legend,
                                  edgecolor=edgecolor, facecolor=facecolor, alpha=alpha)

        _plt.tight_layout()

        if probes:
            _ = [axes[0].plot(self.x[i], self.z[j], 'ro') for i, j in probes]
            _ = [axes[1].plot(self.xp[i, j], self.zp[i, j], 'ro') for i, j in probes]

        if pml:
            _graphics.plot_pml(axes[0], self.x, self.z, self.bc, self.Npml)
            _graphics.plot_pml(axes[1], self.xp, self.zp, self.bc, self.Npml)

        if bc_profiles:
            _graphics.plot_bc_profiles(axes[0], self.x, self.z, self.obstacles)
            _graphics.plot_bc_profiles(axes[1], self.xp, self.zp, self.obstacles)

        for ax in axes:
            ax.set_aspect('equal')
            ax.set_xlabel(r'x [m]')
            ax.set_ylabel(r'z [m]')


        if isinstance(filename, str):
            _plt.savefig(filename)

    def _make_grid(self):

        # Coordinates
        self.x = (_np.arange(self.nx, dtype=float) - self.ix0)*self.dx
        self.z = (_np.arange(self.nz, dtype=float) - self.iz0)*self.dz

        # Numerical coordinates
        self.xn, self.zn = _np.meshgrid(self.x, self.z)
        self.xn = _np.ascontiguousarray(self.xn.T)
        self.zn = _np.ascontiguousarray(self.zn.T)

        # Pysical coordinates
        self.xp, self.zp = self.fcurvxz(self.xn, self.zn)

        # Inverse Jacobian matrix coefficients
        du = _drv.du(_np.arange(self.nx, dtype=float),
                     _np.arange(self.nz, dtype=float), stencil=11, add=False)

        dxp_dxn = du.dudx(self.xp)/du.dudx(self.xn)
        dzp_dxn = du.dudx(self.zp)/du.dudx(self.xn)
        dzp_dzn = du.dudz(self.zp)/du.dudz(self.zn)
        dxp_dzn = du.dudz(self.xp)/du.dudz(self.zn)

        # Jacobian matrix coefficients
        self.J = 1/(dxp_dxn*dzp_dzn - dxp_dzn*dzp_dxn)
        self.dxn_dxp = self.J*dzp_dzn
        self.dzn_dzp = self.J*dxp_dxn
        self.dxn_dzp = -self.J*dxp_dzn
        self.dzn_dxp = -self.J*dzp_dxn

        # Check GCL
        gcl_x = du.dudx(self.dxn_dxp/self.J) + du.dudz(self.dzn_dxp/self.J)
        gcl_z = du.dudx(self.dxn_dzp/self.J) + du.dudz(self.dzn_dzp/self.J)
        if _np.abs(gcl_x).max() > 1e-8 or _np.abs(gcl_z).max() > 1e-8:
            print('GCL (x) : ', _np.abs(gcl_x).max())
            print('GCL (z) : ', _np.abs(gcl_z).max())
            raise _exceptions.GridError('Geometric Conservation Laws not verified')

    def _make_moving_bc(self):

        for obs in self.obstacles:
            obs.make_moving_bc(self.x, self.z)

    def __str__(self):
        s = 'Curvilinear {}x{} points grid with {} boundary conditions:\n\n'
        s += '\t* Spatial step  : {}\n'.format((self.dx, self.dz))
        s += '\t* Origin        : {}\n'.format((self.ix0, self.iz0))
        if 'A' in self.bc:
            s += '\t* Points in PML : {}\n'.format(self.Npml)
        s += '\t* Max stencil   : {}\n'.format(self.stencil)

        return s.format(self.nx, self.nz, self.bc)


if __name__ == "__main__":

    import templates

    shp = 256, 128
    stps = 1e-4, 1e-4
    orgn = 128, 64
    obstcle = templates.plus(*shp)


    mesh = Mesh(shp, stps, orgn, obstacles=obstcle, bc='PAPZ')
    mesh.plot_domains(legend=True, N=2)
