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

Submodule `mesh` provides the mesh class.

@author: Cyril Desjouy
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, ticker
from ofdlib2 import derivation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .utils import down_sample
from .cdomain import ComputationDomains
from .domains import Subdomain
from .templates import curv


def plot_obstacles(x, z, ax, obstacles, facecolor='k', edgecolor='r', curvilinear=False):
    """ plot all obstacles in ax. obstacle is a list of all coordinates. """

    for obs in obstacles:

        if curvilinear and isinstance(obs, Subdomain):
            for i in range(2):
                ax.plot(x[obs.rx, obs.iz[i]], z[obs.rx, obs.iz[i]],
                        color=edgecolor, linewidth=3)
                ax.plot(x[obs.ix[i], obs.rz], z[obs.ix[i], obs.rz],
                        color=edgecolor, linewidth=3)

        elif curvilinear and isinstance(obs, np.ndarray):
            ax.plot(x[obs[0]:obs[2]+1, obs[1]], z[obs[0]:obs[2]+1, obs[1]],
                    color=edgecolor, linewidth=3)
            ax.plot(x[obs[0]:obs[2]+1, obs[3]], z[obs[0]:obs[2]+1, obs[3]],
                    color=edgecolor, linewidth=3)

            ax.plot(x[obs[0], obs[1]:obs[3]+1], z[obs[0], obs[1]:obs[3]+1],
                    color=edgecolor, linewidth=3)
            ax.plot(x[obs[2], obs[1]:obs[3]+1], z[obs[2], obs[1]:obs[3]+1],
                    color=edgecolor, linewidth=3)

        else:
            if isinstance(obs, np.ndarray):
                rect = patches.Rectangle((x[obs[0]], z[obs[1]]),
                                         x[obs[2]] - x[obs[0]],
                                         z[obs[3]] - z[obs[1]],
                                         linewidth=3,
                                         edgecolor=edgecolor,
                                         facecolor=facecolor,
                                         alpha=0.5)
            elif isinstance(obs, Subdomain):
                rect = patches.Rectangle((x[obs.ix[0]], z[obs.iz[0]]),
                                         x[obs.ix[1]] - x[obs.ix[0]],
                                         z[obs.iz[1]] - z[obs.iz[0]],
                                         linewidth=3,
                                         edgecolor=edgecolor,
                                         facecolor=facecolor,
                                         alpha=0.5)
            ax.add_patch(rect)


class BoundaryConditionError(Exception):
    """ Error due to incompatible boundary conditions """
    pass


class GridError(Exception):
    """ Error due wrong grid parameters. """
    pass


class Mesh:
    """ Mesh Class : Construct grid with constant spatial steps

    Parameters
    ----------

    shape : Size of the domain. Must be a 2 elements tuple of int
    step : Spatial steps. Must be a 2 elements tuple of float
    origin : Origin of the grid. Must be a 2 elements tuple of int. Optional
    bc : Boundary conditions. Must be a string of 4 chacracter among 'A', 'R', 'Z' and 'P'
    obstacles : Domain object. Set of obstacles to consider
    Npml : Int. Number of points of the absorbing area (if 'A' is in bc). Optional
    stencil : Int. Size of the finite difference stencil that will be used by nsfds2
    """

    def __init__(self, shape, step, origin=(0, 0),
                 bc='RRRR', obstacles=None, Npml=15, stencil=11):


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

        self._make_grid()
        self._find_subdomains()

    def _make_grid(self):
        """ Make grid. """

        self.x = (np.arange(self.nx)-self.ix0)*self.dx
        self.z = (np.arange(self.nz)-self.iz0)*self.dz

    def N(self, axis):
        """ Return number of point following 'axis'. """
        return self.nx if axis == 0 else self.nz if axis == 1 else None

    def du(self, axis):
        """ Return base spacial step following 'axis'. """
        return self.dx if axis == 0 else self.dz if axis == 1 else None

    def _limits(self):
        xlim = ComputationDomains.bounds(self.shape, self.bc, self.obstacles, axis=0)
        zlim = ComputationDomains.bounds(self.shape, self.bc, self.obstacles, axis=1)
        return xlim, zlim

    def _find_subdomains(self):
        """ Divide the computation domain in subdomains. """

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

    def plot_domains(self, legend=False, N=4, size=(9, 18)):
        """ Plot a scheme of the computation domain higlighting the subdomains.

            Parameters
            ----------
            legend: Show mesh legend
            N: Keep one point over N
        """

        _, axes = plt.subplots(2, 1, figsize=size)

        offset = 10

        # Grid & Obstacles
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

            self._plot_subdomains(ax, self.obstacles, fcolor='k', ecolor='k')

        # Subdomains
        for tag, color in zip(['X', 'W', 'A'], ['b', 'r', 'g']):
            self._plot_subdomains(axes[0], [s for s in self.dxdomains if s.tag == tag],
                                  fcolor='y', ecolor=color, legend=legend)
            self._plot_subdomains(axes[1], [s for s in self.dzdomains if s.tag == tag],
                                  fcolor='y', ecolor=color, legend=legend)
            self._plot_subdomains(axes[0], [s for s in self.adomains if s.tag == tag],
                                  fcolor='y', ecolor=color, legend=legend)
            self._plot_subdomains(axes[1], [s for s in self.adomains if s.tag == tag],
                                  fcolor='y', ecolor=color, legend=legend)

        if legend:
            self._show_bc(axes, offset)
            print(self.domain)

        plt.tight_layout()

    def _show_bc(self, axes, offset):
        """ Show text indicating each of the 4 bc of the domain. """

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

    def _plot_subdomains(self, ax, domain, fcolor='k', ecolor='k', legend=False):

        for sub in domain:
            rect = patches.Rectangle((self.x[sub.ix[0]], self.z[sub.iz[0]]),
                                     self.x[sub.ix[1]]-self.x[sub.ix[0]],
                                     self.z[sub.iz[1]]-self.z[sub.iz[0]],
                                     linewidth=3,
                                     edgecolor=ecolor,
                                     facecolor=fcolor,
                                     alpha=0.5)
            ax.add_patch(rect)

            if legend and sub.tag in ['X', 'A', 'W']:
                ax.text(self.x[sub.center[0]]-self.dx*len(sub.tag)*6,
                        self.z[sub.center[1]]-self.dz*3, sub.key, color=ecolor)

    def plot_grid(self, N=4):
        """ Grid representation """


        nullfmt = ticker.NullFormatter()         # no labels

        fig, ax_c = plt.subplots(figsize=(9, 5))
        fig.subplots_adjust(.1, .1, .95, .95)

        divider = make_axes_locatable(ax_c)
        ax_xa = divider.append_axes('top', size='30%', pad=0.1)
        ax_xb = divider.append_axes('top', size='20%', pad=0.1)
        ax_za = divider.append_axes('right', size='15%', pad=0.1)
        ax_zb = divider.append_axes('right', size='10%', pad=0.1)

        self._plot_subdomains(ax_c, self.obstacles, fcolor='k')

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

    def get_obstacles(self):
        """ Get a list of the coordinates of all obstacles. """
        return [sub.xz for sub in self.obstacles]

    @staticmethod
    def show_figures():
        """ Show all figures. """
        plt.show()

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
    """ AdaptativeMesh Class : Construct grid with adaptative spatial steps

    Parameters
    ----------

    shape : Size of the domain. Must be a 2 elements tuple of int
    step : Spatial steps. Must be a 2 elements tuple of float
    origin : Origin of the grid. Must be a 2 elements tuple of int. Optional
    bc : Boundary conditions. Must be a string of 4 chacracter among 'A', 'R', 'Z' and 'P'
    obstacles : Domain object. Set of obstacles to consider
    Npml : Int. Number of points of the absorbing area (if 'A' is in bc). Optional
    stencil : Int. Size of the finite difference stencil that will be used by nsfds2
    """

    def _make_grid(self):

        self.x = np.zeros(self.nx)
        self.z = np.zeros(self.nz)

        self.Nr = 46
        self._alpha_r = 2

        self._rr = np.exp(np.log(self._alpha_r)/self.Nr)
        self._ra = np.exp(np.log(3)/self.Npml)

        xlim, zlim = self._limits()

        self.x = self._make_axis(self.x, self.dx, xlim, axis=0)
        self.z = self._make_axis(self.z, self.dz, zlim, axis=1)

        self.x -= self.x[self.ix0]
        self.z -= self.z[self.iz0]

    def _bc_at(self, side, axis):
        """ Return bc following 'axis' at 'side' location. """
        if side == 'start':
            bc = self.bc[axis]
        elif side == 'end':
            bc = self.bc[axis+2]
        return bc

    def _make_axis(self, u, du, ulim, axis):

        for start, stop, _ in ulim:

            if start == 0:
                u, du = self._make_axis_start(u, du, start, stop+1, axis)
            elif stop == self.N(axis) - 1:
                u, du = self._make_axis_end(u, du, start, stop+1, axis)
            else:
                u, du = self._make_in_axis(u, du, start, stop+1, axis)

        return u

    def _make_in_axis(self, u, du, start, stop, axis):

        lim1 = min(start + self.Nr + self.stencil, start + int((stop - start)/2))
        lim2 = max(lim1, stop - self.Nr - self.stencil)

        for i in range(start, stop):

            u[i] = u[i-1] + du

            if start + self.stencil < i < lim1 and du < self.du(axis):
                du *= self._rr
            elif lim2 <= i < stop - self.stencil and du > self.du(axis)/self._alpha_r:
                du /= self._rr

        return u, du

    def _make_axis_start(self, u, du, start, stop, axis):

        du0, bc = self.du(axis), self._bc_at('start', axis)

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

        du0, bc = self.du(axis), self._bc_at('end', axis)

        lim1 = min(start + self.Nr + self.stencil, start + int((stop - start)/2))

        if bc in ['R', 'Z']:
            lim2 = max(lim1, stop - self.Nr - self.stencil)
        elif bc == 'A':
            lim2 = max(lim1, stop - self.Npml - self.stencil)
        elif bc == 'P':
            lim2 = self.N(axis)

        for i in range(start, stop):
            u[i] = u[i-1] + du
            if start + self.stencil <= i < lim1  and du < du0:
                du *= self._rr
            elif lim2 <= i < stop - self.stencil and bc == 'A':
                du *= self._ra
            elif lim2 <= i < stop - self.stencil and bc != 'A' and du > du0/self._alpha_r:
                du /= self._rr

        return u, du

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
    """ CurvilinearMesh Class : Construct grid using curvilinear coordinates

    Parameters
    ----------

    shape : Size of the domain. Must be a 2 elements tuple of int
    step : Spatial steps. Must be a 2 elements tuple of float
    origin : Origin of the grid. Must be a 2 elements tuple of int. Optional
    bc : Boundary conditions. Must be a string of 4 chacracter among 'A', 'R', 'Z' and 'P'
    obstacles : Domain object. Set of obstacles to consider
    Npml : Int. Number of points of the absorbing area (if 'A' is in bc). Optional
    stencil : Int. Size of the finite difference stencil that will be used by nsfds2
    fcurvxz : function taking as input arguments the numerical coordinates (xn/yn)
              and returning the physical coordinates (xp/zp)
    """

    def __init__(self, shape, step, origin=(0, 0),
                 bc='RRRR', obstacles=None, Npml=15, stencil=11, fcurvxz=curv):

        if not fcurvxz:
            self.fcurvxz = curv
        else:
            self.fcurvxz = fcurvxz

        super().__init__(shape, step, origin, bc, obstacles, Npml, stencil)

    @staticmethod
    def _pmesh(ax, x, z, step=8):
        """ Plot grid lines. """

        nbx = x.shape[0]
        nbz = x.shape[1]
        for i in np.arange(0, nbz, int(nbx/step)):
            ax.plot(x[:, i], z[:, i], 'k')

        ax.plot(x[:, -1], z[:, -1], 'k')

        for i in np.arange(0, nbx, int(nbz/step)):
            ax.plot(x[i, :], z[i, :], 'k')

        ax.plot(x[-1, :], z[-1, :], 'k')

    def plot_curvi(self):
        """ Plot physical and numerical domains. """

        _, axes = plt.subplots(ncols=2, figsize=(9, 3))
        self._pmesh(axes[0], self.xn, self.zn)
        self._pmesh(axes[1], self.xp, self.zp)
        axes[0].set(title='Numerical Domain')
        axes[1].set(title='Physical Domain')

        plot_obstacles(self.x, self.z, axes[0], self.obstacles,
                       edgecolor='k')
        plot_obstacles(self.xp, self.zp, axes[1], self.obstacles,
                       edgecolor='k', curvilinear=True)

        axes[0].set_aspect('equal')
        axes[1].set_aspect('equal')
        plt.tight_layout()
        plt.show()

    def _make_grid(self):

        # Coordinates
        self.x = (np.arange(self.nx, dtype=float) - self.ix0)*self.dx
        self.z = (np.arange(self.nz, dtype=float) - self.iz0)*self.dz

        # Numerical coordinates
        self.xn, self.zn = np.meshgrid(self.x, self.z)
        self.xn = np.ascontiguousarray(self.xn.T)
        self.zn = np.ascontiguousarray(self.zn.T)

        # Pysical coordinates
        self.xp, self.zp = self.fcurvxz(self.xn, self.zn)

        # Inverse Jacobian matrix coefficients
        du = derivation.du11(np.arange(self.nx, dtype=float),
                             np.arange(self.nz, dtype=float), add=False)

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
        if np.abs(gcl_x).max() > 1e-8 or np.abs(gcl_z).max() > 1e-8:
            print('GCL (x) : ', np.abs(gcl_x).max())
            print('GCL (z) : ', np.abs(gcl_z).max())
            raise ValueError('Geometric Conservation Laws not verified')


if __name__ == "__main__":

    import templates

    shp = 256, 128
    stps = 1e-4, 1e-4
    orgn = 128, 64
    obstcle = templates.plus(*shp)


    mesh1 = Mesh(shp, stps, orgn, obstacles=obstcle, bc='PAPZ')
    mesh1.plot_domains(legend=True, N=2)
