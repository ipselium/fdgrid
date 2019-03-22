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
"""
-----------

Submodule `mesh` provides the mesh class.

@author: Cyril Desjouy
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import NullFormatter
from .utils import down_sample
from .domain import Domain


class BoundaryConditionError(Exception):
    """ Error due to incompatible boundary conditions """
    pass


class GridError(Exception):
    """ Error due to incompatible boundary conditions """
    pass


class Mesh:
    """ Mesh Class : Construct meshgrid for finite differences

    bc : 'TPLR' (left, bottom, right, top)

         R : Rigid
         P : Periodic
         A : PML
         Z : Impedance

    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, shape, step, origin=(0, 0), bc='RRRR', obstacles=None, Npml=15, stencil=11):

        # pylint: disable=too-many-arguments

        self.shape = shape
        self.nx, self.nz = shape
        self.dx, self.dz = step
        self.ix0, self.iz0 = origin
        self.stencil = stencil
        self.Npml = Npml

        self.obstacles = obstacles
        self.bc = bc.upper()
        self._check_bc()
        self._check_grid()

        self.one_dx = 1/self.dx
        self.one_dz = 1/self.dz

        self._make_grid()
        self._find_subdomains()

    def _make_grid(self):
        """ Make grid. """

        self.x = (np.arange(self.nx)-self.ix0)*self.dx
        self.z = (np.arange(self.nz)-self.iz0)*self.dz

    def _limits(self):
        xlim = Domain.bounds(self.nx, self.obstacles, 'x')
        zlim = Domain.bounds(self.nz, self.obstacles, 'z')
        return xlim, zlim
        #return Domain.split_bounds(xlim, self.nx), Domain.split_bounds(zlim, self.nz)

    def _find_subdomains(self):
        """ Divide the computation domain in subdomains. """

        self.domain = Domain((self.nx, self.nz), self.obstacles, self.bc, self.stencil)
        self.xdomains, self.zdomains = self.domain.listing
        self.all_domains = self.xdomains + self.zdomains

    def _check_bc(self):

        regex = [r'[^P].P.', r'P.[^P].', r'.[^P].P', r'.P.[^P]']

        if not re.match(r'^[ZRAP]*$', self.bc):
            raise BoundaryConditionError("bc must be combination of 'ZRAP'!")

        if any(re.match(r, self.bc) for r in regex):
            raise BoundaryConditionError("periodic condition must be on both sides of the domain,"
                                         + " i.e. '(P.P.)'|'(.P.P)'")

    def _check_grid(self):

        if self.Npml < self.stencil:
            raise GridError("Number of points of PML must be greater than stencil")

        if self.ix0 > self.nx or self.iz0 > self.nz:
            raise GridError("Origin of the domain must be in the domain")

    def plot_domains(self, annotations=True, N=4):
        """ Plot a scheme of the computation domain higlighting the subdomains.

            Parameters
            ----------
            annotations: Show mesh annotations
            N: Keep one point over N
        """

        _, axes = plt.subplots(2, 1, figsize=(9, 8))

        offset = 10

        for ax in axes:
            ax.set_xlim(self.x.min() - offset*self.dx, self.x.max() + offset*self.dx)
            ax.set_ylim(self.z.min() - offset*self.dz, self.z.max() + offset*self.dz)
            ax.set_xlabel('x [m]')
            ax.set_ylabel('z [m]')
            ax.set_aspect('equal')

            for z in down_sample(self.z, N):
                ax.plot(self.x, z.repeat(self.nx), 'k', linewidth=0.5)

            for x in down_sample(self.x, N):
                ax.plot(x.repeat(self.nz), self.z, 'k', linewidth=0.5)

            self._plot_areas(ax, self.obstacles, fcolor='k')

        self._plot_areas(axes[0], self.xdomains, fcolor='y', annotations=annotations)
        self._plot_areas(axes[1], self.zdomains, fcolor='y', annotations=annotations)

        if annotations:

            xloc = np.array([self.x[0] - self.dx*offset/2,
                             (self.x[-1] - np.abs(self.x[0]))/2 + self.dx,
                             self.x[-1] + self.dx,
                             (self.x[-1] - np.abs(self.x[0]))/2])

            zloc = np.array([(self.z[-1] - np.abs(self.z[0]))/2,
                             self.z[0] - self.dz*offset/1.25,
                             (self.z[-1] - np.abs(self.z[0]))/2,
                             self.z[-1] + self.dz])

            for bc, x, z in zip(self.bc, xloc, zloc):
                for ax in axes:
                    ax.text(x, z, bc, color='r')

            print(self.domain)

        plt.tight_layout()
        plt.show()

    def _plot_areas(self, ax, area, fcolor='k', annotations=True):

        for a in area:
            rect = patches.Rectangle((self.x[a.ix[0]], self.z[a.iz[0]]),
                                     self.x[a.ix[1]]-self.x[a.ix[0]],
                                     self.z[a.iz[1]]-self.z[a.iz[0]],
                                     linewidth=3, edgecolor='r', facecolor=fcolor, alpha=0.5)
            ax.add_patch(rect)

            if annotations:
                ax.text(self.x[a.ix[0]+2], self.z[a.iz[0]+2], a.no, color='b')

    @staticmethod
    def plot_obstacles(x, z, ax, obstacles, facecolor='k', edgecolor='r'):
        """ plot all obstacles in ax. obstacle is a list of all coordinates. """

        for obs in obstacles:
            rect = patches.Rectangle((x[obs[0]], z[obs[1]]),
                                     x[obs[2]] - x[obs[0]],
                                     z[obs[3]] - z[obs[1]],
                                     linewidth=3,
                                     edgecolor=edgecolor,
                                     facecolor=facecolor,
                                     alpha=0.5)
            ax.add_patch(rect)

    def plot_grid(self, N=4):
        """ Grid representation """


        nullfmt = NullFormatter()         # no labels

        fig, ax_c = plt.subplots(figsize=(9, 5))
        fig.subplots_adjust(.1, .1, .95, .95)

        divider = make_axes_locatable(ax_c)
        ax_xa = divider.append_axes('top', size='30%', pad=0.1)
        ax_xb = divider.append_axes('top', size='20%', pad=0.1)
        ax_za = divider.append_axes('right', size='15%', pad=0.1)
        ax_zb = divider.append_axes('right', size='10%', pad=0.1)

        self._plot_areas(ax_c, self.obstacles, fcolor='k')

        for z in down_sample(self.z, N):
            ax_c.plot(self.x, z.repeat(self.nx), 'k', linewidth=0.5)

        for x in down_sample(self.x, N):
            ax_c.plot(x.repeat(self.nz), self.z, 'k', linewidth=0.5)


        ax_c.set_xlim(self.x.min(), self.x.max())
        ax_c.set_ylim(self.z.min(), self.z.max())
        ax_c.set_xlabel(r'$x$ [m]')
        ax_c.set_ylabel(r'$z$ [m]')
        ax_c.set_aspect('equal')

        ax_xa.plot(self.x[:-1], np.diff(self.x)/self.dx, 'ko')
        ax_xa.set_xlim(ax_c.get_xlim())
        ax_xa.set_ylim((np.diff(self.x)/self.dx).min()/1.5, (np.diff(self.x)/self.dx).max()*1.5)
        ax_xa.xaxis.set_major_formatter(nullfmt)  # no label
        ax_xa.set_ylabel(r"$x'/dx$")
        ax_xb.plot(self.x, range(len(self.x)), 'k', linewidth=2)
        ax_xb.set_xlim(ax_c.get_xlim())
        ax_xb.xaxis.set_major_formatter(nullfmt)  # no label
        ax_xb.set_ylabel(r"$N_x$")

        ax_za.plot(np.diff(self.z)/self.dz, self.z[:-1], 'ko')
        ax_za.set_ylim(ax_c.get_ylim())
        ax_za.set_xlim((np.diff(self.z)/self.dz).min()/1.5, (np.diff(self.z)/self.dz).max()*1.5)
        ax_za.yaxis.set_major_formatter(nullfmt)  # no label
        ax_za.set_xlabel(r"$z'/dz$")
        ax_zb.plot(range(len(self.z)), self.z, 'k', linewidth=2)
        ax_zb.set_ylim(ax_c.get_ylim())
        ax_zb.yaxis.set_major_formatter(nullfmt)  # no label
        ax_zb.set_xlabel(r"$N_z$")

        for j in self._limits()[0]:
            if j[-1] == 'o':
                ax_xa.axvspan(self.x[j[0]], self.x[j[1]], facecolor='k', alpha=0.5)
                ax_xb.axvspan(self.x[j[0]], self.x[j[1]], facecolor='k', alpha=0.5)
        for j in self._limits()[1]:
            if j[-1] == 'o':
                ax_za.axhspan(self.z[j[0]], self.z[j[1]], facecolor='k', alpha=0.5)
                ax_zb.axhspan(self.z[j[0]], self.z[j[1]], facecolor='k', alpha=0.5)

    def get_obstacles(self):
        """ Get a list of the coordinates of all obstacles. """
        return [sub.xz for sub in self.obstacles]

    def __str__(self):
        s = 'Cartesian {}x{} points grid with {} boundary conditions:\n\n'
        s += '\t* Spatial step : {}\n'.format((self.dx, self.dz))
        s += '\t* Origin       : {}\n'.format((self.ix0, self.iz0))
        s += '\t* Points in PML: {}\n'.format(self.Npml)
        s += '\t* Max stencil   : {}\n'.format(self.stencil)

        return s.format(self.nx, self.nz, self.bc)

    def __repr__(self):
        return self.__str__()


class AdaptativeMesh(Mesh):
    """ Mesh Class : Construct meshgrid for finite differences with adaptative grid

    bc : 'TPLR' (left, bottom, right, top)

         R : Rigid
         P : Periodic
         A : PML
         Z : Impedance

    """

    def _make_grid(self):

        self.x = np.zeros(self.nx)
        self.z = np.zeros(self.nz)

        self.Nr = 46
        self._alpha_r = 2

        self._rr = np.exp(np.log(self._alpha_r)/self.Nr)
        self._ra = np.exp(np.log(3)/self.Npml)

        xlim, zlim = self._limits()

        self.x = self._make_axis(self.x, self.dx, xlim, axis='x')
        self.z = self._make_axis(self.z, self.dz, zlim, axis='z')

        self.x -= self.x[self.ix0]
        self.z -= self.z[self.iz0]

    def N(self, axis):
        """ Return number of point following 'axis'. """
        return self.nx if axis == 'x' else self.nz if axis == 'z' else None

    def du(self, axis):
        """ Return base spacial step following 'axis'. """
        return self.dx if axis == 'x' else self.dz if axis == 'z' else None

    def bc_at(self, side, axis):
        """ Return bc following 'axis' at 'side' location. """
        if axis == 'x':
            if side == 'start':
                bc = self.bc[0]
            elif side == 'end':
                bc = self.bc[2]
        elif axis == 'z':
            if side == 'start':
                bc = self.bc[1]
            elif side == 'end':
                bc = self.bc[3]
        return bc

    def _make_axis(self, u, du, ulim, axis):

        for start, stop, _ in ulim:

            if start == 0:
                u, du = self._make_axis_start(u, du, start, stop, axis)
            elif stop == self.N(axis) - 1:
                u, du = self._make_axis_end(u, du, start, stop, axis)
            else:
                u, du = self._make_in_axis(u, du, start, stop, axis)

        return u

    def _make_in_axis(self, u, du, start, stop, axis):

        lim1 = min(start + self.Nr + self.stencil, start + int((stop - start)/2))
        lim2 = max(lim1, stop - self.Nr - self.stencil)

        for i in range(start, stop):

            u[i+1] = u[i] + du

            if start + self.stencil < i < lim1 and du < self.du(axis):
                du *= self._rr
            elif lim2 <= i < stop - self.stencil and du > self.du(axis)/self._alpha_r:
                du /= self._rr

        return u, du

    def _make_axis_start(self, u, du, start, stop, axis):

        du0, bc = self.du(axis), self.bc_at('start', axis)

        if bc == 'P':
            lim1 = 0
        elif bc in ['R', 'Z']:
            lim1 = min(self.Nr+self.stencil, int((stop - start)/2))
            du = du0/self._alpha_r
        elif bc == 'A':
            lim1 = min(self.Npml, int((stop - start)/2))
            for i in range(lim1):
                du *= self._ra

        lim2 = max(lim1, stop - self.Nr)

        for i in range(start, stop):
            u[i+1] = u[i] + du
            if self.stencil <= i < lim1 and bc != 'A' and du < du0:
                du *= self._rr
            elif self.stencil < i < lim1 and bc == 'A':
                du /= self._ra
            elif lim2 <= i < stop - self.stencil and bc != 'A' and du > du0/self._alpha_r:
                du /= self._rr

        return u, du

    def _make_axis_end(self, u, du, start, stop, axis):

        du0, bc = self.du(axis), self.bc_at('end', axis)

        lim1 = min(start + self.Nr + self.stencil, start + int((stop - start)/2))

        if bc in ['R', 'Z']:
            lim2 = max(lim1, stop - self.Nr - self.stencil)
        elif bc == 'A':
            lim2 = max(lim1, stop - self.Npml - self.stencil)
        elif bc == 'P':
            lim2 = self.N(axis)

        for i in range(start, stop):
            u[i+1] = u[i] + du
            if start + self.stencil <= i < lim1  and du < du0:
                du *= self._rr
            elif lim2 <= i < stop - self.stencil and bc == 'A':
                du *= self._ra
            elif lim2 <= i < stop - self.stencil and bc != 'A' and du > du0/self._alpha_r:
                du /= self._rr

        return u, du

    def plot_xz(self):
        """ Plot mesh """

        _, axes = plt.subplots(2, 2, figsize=(9, 4))
        axes[0, 0].plot(self.x, 'k*')
        axes[0, 0].set_xlabel(r'$N_x$')
        axes[0, 0].set_ylabel(r'$x$ [m]')

        axes[1, 0].plot(np.diff(self.x)/self.dx, 'k*')
        axes[1, 0].set_xlabel(r'$x$ [m]')
        axes[1, 0].set_ylabel(r"$x'/dx$")

        axes[0, 1].plot(self.z, 'k*')
        axes[0, 1].set_xlabel(r'$N_z$')
        axes[0, 1].set_ylabel(r'$z$ [m]')

        axes[1, 1].plot(np.diff(self.z)/self.dz, 'k*')
        axes[1, 1].set_xlabel(r'$z$ [m]')
        axes[1, 1].set_ylabel(r"$z'/dz$")

        for i in range(axes.shape[1]):
            for ax in axes[:, i]:
                ax.grid()
                for j in self._limits()[i]:
                    if j[-1] == 'o':
                        ax.axvspan(j[0], j[1], facecolor='k', alpha=0.5)

        plt.tight_layout()
        plt.show()

    def __str__(self):
        s = 'Adaptative cartesian {}x{} points grid with {} boundary conditions:\n\n'
        s += '\t* Spatial step  : {}\n'.format((self.dx, self.dz))
        s += '\t* Origin        : {}\n'.format((self.ix0, self.iz0))
        s += '\t* Points in PML : {}\n'.format(self.Npml)
        s += '\t* Max stencil   : {}\n'.format(self.stencil)
        s += '\t* Dilation obs  : {:.1f}%\n'.format((self._rr-1)*100)
        s += '\t* Dilation pml  : {:.1f}%\n'.format((self._ra-1)*100)

        return s.format(self.nx, self.nz, self.bc)


class CurvilinearMesh(Mesh):
    """ Curvilinear Mesh """
    pass


if __name__ == "__main__":

    import templates

    shape = (256, 128)
    steps = (1e-4, 1e-4)
    origin = 128, 64
    obstacle = templates.plus(*shape)


    mesh1 = Mesh(shape, steps, origin, obstacles=obstacle, bc='PAPZ')
    mesh1.plot_domains(annotations=True, N=2)
