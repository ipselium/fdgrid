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
# Creation Date : 2019-02-13 - 07:54:54
# pylint: disable=too-many-instance-attributes
# pylint: disable=len-as-condition
# pylint: disable=pointless-statement
"""
-----------

Module `cdomain` provides `ComputationDomains` class aiming at dividing the
grid into Subdomain objects used by nsfds2

-----------
"""


import re as _re
import copy as _copy
import warnings as _warnings
import itertools as _itertools
import numpy as _np
import matplotlib.pyplot as _plt
from fdgrid import _exceptions
from fdgrid import utils as _utils
from fdgrid.domains import Domain, Subdomain


class ComputationDomains:
    """ Divide computation domain in several Subdomains based on obstacles in presence.

    Parameters
    ----------
    shape : tuple
        Size of the domain. Must be a tuple with two int objects.
    obstacles : :py:class:`fdgrid.domains.Domain`, optional
        Obstacles in the computation domain.
    bc : {'[ARZPW][ARZPW][ARZPW][ARZPW]'}, optional
        Boundary conditions. Must be a 4 characters string.
    Npml : int, optional
        Number of points of the absorbing area (only if 'A' in `bc`).
    stencil : int, optional
        Size of the finite difference stencil (used by :py:mod:`nsfds2`).

    Raises
    ------
    CloseObstaclesError: when obstacles are too close from each other or from
    the grid boundary

    """

    def __init__(self, shape, obstacles, bc, stencil, Npml=15):

        self._shape = shape
        self._obstacles = obstacles
        self._check_obstacles()
        self._bc = bc
        self._stencil = stencil
        self._Npml = Npml
        self._nx, self._nz = shape
        self.dxdomains = Domain(shape)     # For x derivatives
        self.dzdomains = Domain(shape)     # For z derivatives
        self.fxdomains = Domain(shape)    # For x filters
        self.fzdomains = Domain(shape)    # For z filters
        self.adomains = Domain(shape)     # For PMLs
        self.gdomain = Subdomain([0, 0, self._nx-1, self._nz-1],
                                 bc=self._bc, key=0, tag='g')

        # Indexes of the subdomains
        self._tnd = 0
        self._tndx = 0
        self._tndz = 0

        # Initialize masks
        self._xmask, self._zmask = self.init_masks(shape, bc, obstacles)

        # Set up all Domain and Subdomain objects
        self._proceed()

    def _proceed(self):
        """ Proceed to domain division. """

        # Search computation domains
        self._find_domains()

        # Check if domain is completely processed
        self._check_subdomains()

        # join Subdomains that can be joined
        self.dxdomains.autojoin()
        self.dzdomains.autojoin()

        # Update Domain objects considering periodic bc
        if _re.match(r'P.P.', self._bc):
            mask, _ = self.init_masks(self._shape, self._bc, self._obstacles)
            self.dxdomains.periodic_bc(mask, axis=0)
        if _re.match(r'.P.P', self._bc):
            mask, _ = self.init_masks(self._shape, self._bc, self._obstacles)
            self.dzdomains.periodic_bc(mask, axis=1)

        # Fill filter Subdomains
        self.fxdomains = _copy.deepcopy(self.dxdomains)
        self.fzdomains = _copy.deepcopy(self.dzdomains)

        # Check compatibility between obstacles and PML and determine PML Domain
        if 'A' in self._bc:
            self._check_pml()
            self._fdomain_bc()
            self._built_pml()

        if 'R' in self._bc:
            pass

        # Check if Subdomain are larger enough for stencil
        self._check_stencil()

        # Sort domains
        self.dxdomains.sort(inplace=True)
        self.dzdomains.sort(inplace=True)
        self.adomains.sort(inplace=True)

    def _check_obstacles(self):
        """ check if obstacles bc are 'ZWV'. """

        for obs in self._obstacles:
            if not _re.match(r'^[ZWV]+$', obs.bc):
                msg = "Obstacle bc must be combination of 'ZWV'"
                raise _exceptions.BoundaryConditionError(msg)

    def _fdomain_bc(self):
        """ Change all 'A' for 'W' in filter domains. """

        for fdomain in [self.fxdomains, self.fzdomains]:
            for sub in fdomain:
                if 'A' in sub.bc:
                    sub.bc = sub.bc.replace('A', 'W')

    def _check_stencil(self):
        """ Check if Subdomain are larger enough considering the stencil. """

        msg = "{} too small considering stencil"

        for sub in self.dxdomains:
            if len(sub.rx) < 2*self._stencil + 1 and sub.bc == "WW":
                raise _exceptions.CloseObstaclesError(msg.format(sub))

        for sub in self.dzdomains:
            if len(sub.rz) < 2*self._stencil + 1 and sub.bc == "WW":
                raise _exceptions.CloseObstaclesError(msg.format(sub))

        for sub in self.adomains:
            if sub.axis == 0 and len(sub.rz) < 2*self._stencil and _re.match(r'.W.W', sub.bc):
                raise _exceptions.CloseObstaclesError(msg.format(sub))

            if sub.axis == 1 and len(sub.rx) < 2*self._stencil and _re.match(r'W.W.', sub.bc):
                raise _exceptions.CloseObstaclesError(msg.format(sub))

    def _built_pml(self):

        # Split PML
        self._split_pml(self.dxdomains)
        self._split_pml(self.dzdomains)

        # Join PML
        self._autojoin_pml(self.dxdomains)
        self._autojoin_pml(self.dzdomains)

        # Construct PML
        self.adomains = self._make_pml_domain(self.dxdomains, self.dzdomains)
        self.adomains.reset_keys()

    def _split_pml(self, tosplit):
        """ Split borders of the Domain in Subdomains when PML and tag them. """

        for i in range(4):
            if self.gdomain.bc[i] == 'A':
                axis = 0 if i%2 else 1
                for sub in [s for s in tosplit if s.xz[i] == self.gdomain.xz[i]]:
                    # adjust the split as function of PML position :
                    corr = 1 if sub.xz[i] == 0 else 0
                    tosplit.split(sub.key, abs(self.gdomain.xz[i]-self._Npml+corr), axis=axis)

        # Update tags
        for sub in tosplit:
            idx = _np.argwhere(_np.array(sub.xz) - _np.array(self.gdomain.xz) == 0).ravel()
            for i in idx:
                if self.gdomain.bc[i] == 'A' and self._Npml in (len(sub.rx), len(sub.rz)):
                    if sub.tag != 'A':
                        sub.tag = 'A'
                    else:
                        sub.tag = 'Ac'

    @staticmethod
    def _autojoin_pml(tojoin):
        """ Join pml Subdomains that can be. """

        while True:

            lst = []
            dom = [s for s in tojoin if s.tag == 'A']

            for sub1, sub2 in _itertools.combinations(dom, r=2):
                if tojoin.isjoinable(sub1, sub2, include_split=True):
                    lst.append(sub1.key)
                    lst.append(sub2.key)
                    break

            if lst:
                tojoin.join(*lst, include_split=True)
            else:
                break

    def _make_pml_domain(self, *args):
        """ Make the PML Domain object. """

        # Transfer PML subdomain of xdomains and zdomains to adomains
        tmp = Domain(self._shape)
        for domains in args:
            l = [s for s in domains if s.tag[0] == 'A']
            for sub in l:
                tmp.steal(domains, sub.key)

        # Remove duplicates and update bc
        newsubs = Domain(self._shape)
        for sub1, sub2 in _itertools.combinations(tmp, r=2):
            if sub1.xz == sub2.xz:
                bc = _utils.merge_bc(sub1.bc, sub2.bc)
                sub1.bc = bc
                sub1.key = sub1.key.replace('x', 'a').replace('z', 'a')
                newsubs.append(sub1)

        # Update tag and axis
        for sub in newsubs:
            if sub.tag == 'Ac':
                sub.tag = 'A'
                sub.axis = 2
            elif len(sub.rx) == self._Npml and (sub.ix[0] == 0 or sub.ix[1] == self._nx-1):
                sub.axis = 0
            elif len(sub.rz) == self._Npml and (sub.iz[0] == 0 or sub.iz[1] == self._nz-1):
                sub.axis = 1


        return newsubs

    def _check_pml(self):
        """ Check if PML is compatible with obstacles. """

        msg = 'Obstacle {} not compatible with PML'

        for sub in self._obstacles:
            if self._bc[0] == 'A':
                if 0 < sub.xz[0] < self._Npml + self._stencil or sub.xz[2] < self._Npml+1:
                    raise _exceptions.CloseObstaclesError(msg.format(sub))

            if self._bc[1] == 'A':
                if 0 < sub.xz[1] < self._Npml + self._stencil or sub.xz[3] < self._Npml+1:
                    raise _exceptions.CloseObstaclesError(msg.format(sub))

            if self._bc[2] == 'A':
                if self._nx-self._Npml-self._stencil < sub.xz[2] < self._nx-1 \
                        or sub.xz[0] >= self._nx - self._Npml - 1:
                    raise _exceptions.CloseObstaclesError(msg.format(sub))

            if self._bc[3] == 'A':
                if self._nz-self._Npml-self._stencil < sub.xz[3] < self._nz-1 \
                        or sub.xz[1] >= self._nz - self._Npml - 1:
                    raise _exceptions.CloseObstaclesError(msg.format(sub))

    def _find_domains(self):
        """ Find computation domains. """
        for obs in self._obstacles:
            if obs.ix[1] != self._nx-1:
                self._right(obs)
            if obs.ix[0] != 0:
                self._left(obs)
            if obs.iz[1] != self._nz-1:
                self._top(obs)
            if obs.iz[0] != 0:
                self._bottom(obs)

        self._xbetween()
        self._zbetween()
        self._fill()
        self._update_patches()

    def _check_subdomains(self):
        """ Make some tests to check if all subdomains are well defined. """
        if not self.is_domain_complete:
            msg = "Uncomplete computation domain. See domains.show_missings() method."
            _warnings.warn(msg)
        if not self.is_bc_complete:
            msg = "Undefined boundary for {}. Check subdomains."
            _warnings.warn(msg.format(self.missing_bc[0].key))

    def _top(self, obs):
        s = self._zmask[obs.sx, obs.iz[1]+1]
        tmp = _np.argwhere(s == 0).ravel()
        if len(tmp) > 0:
            lst = _utils.split_discontinuous(tmp + obs.ix[0])
            for idx in lst:
                self._extrude_top(obs, idx)

    def _bottom(self, obs):
        s = self._zmask[obs.sx, obs.iz[0]-1]
        tmp = _np.argwhere(s == 0).ravel()
        if len(tmp) > 0:
            lst = _utils.split_discontinuous(tmp + obs.ix[0])
            for idx in lst:
                self._extrude_bottom(obs, idx)

    def _right(self, obs):
        s = self._xmask[obs.ix[1]+1, obs.sz]
        tmp = _np.argwhere(s == 0).ravel()
        if len(tmp) > 0:
            lst = _utils.split_discontinuous(tmp + obs.iz[0])
            for idz in lst:
                self._extrude_right(obs, idz)

    def _left(self, obs):
        s = self._xmask[obs.ix[0]-1, obs.sz]
        tmp = _np.argwhere(s == 0).ravel()
        if len(tmp) > 0:
            lst = _utils.split_discontinuous(tmp + obs.iz[0])
            for idz in lst:
                self._extrude_left(obs, idz)

    def _extrude_top(self, obs, idx):
        """ Extrude top wall of the obstacle. """
        xc1, zc1 = idx[0], obs.xz[3]
        xc2, zc2 = idx[1], obs.xz[3]
        bc = '.{}.{}'.format(obs.bc[3], self._bc[3])
        for j in range(zc1+1, self._nz):
            zc2 = j
            mk = self._zmask[xc1+1:xc2, j]
            if _np.any(mk != 0):
                if _np.all(mk == -2):
                    bc = '.{}.W'.format(obs.bc[3])
                elif _np.any(mk == -2):
                    zc2 = int((zc1+zc2)/2 - 1)
                    bc = '.{}.X'.format(obs.bc[3])
                elif _np.any(mk == 1):
                    zc2 -= 1
                    bc = '.{}.X'.format(obs.bc[3])
                break

        if abs(zc2 - zc1) < self._stencil:
            msg = _exceptions.CloseObstaclesError.msg.format(obs, self._stencil)
            raise _exceptions.CloseObstaclesError(msg)

        self._update_domains(Subdomain((xc1, zc1, xc2, zc2), bc, self._ndz, axis=1, tag='X'))

    def _extrude_bottom(self, obs, idx):
        """ Extrude bottom wall of the obstacle. """
        xc1, zc1 = idx[0], obs.xz[1]
        xc2, zc2 = idx[1], obs.xz[1]
        bc = '.{}.{}'.format(self._bc[1], obs.bc[1])
        for j in range(zc1-1, -1, -1):
            zc2 = j
            mk = self._zmask[xc1+1:xc2, j]
            if _np.any(mk != 0):
                if _np.all(mk == -2):
                    bc = '.W.{}'.format(obs.bc[1])
                elif _np.any(mk == -2):
                    zc2 = int((zc1+zc2)/2 + 1)
                    bc = '.X.{}'.format(obs.bc[1])
                elif _np.any(mk == 1):
                    zc2 += 1
                    bc = '.X.{}'.format(obs.bc[1])
                break

        if abs(zc2 - zc1) < self._stencil:
            msg = _exceptions.CloseObstaclesError.msg.format(obs, self._stencil)
            raise _exceptions.CloseObstaclesError(msg)

        self._update_domains(Subdomain((xc1, zc2, xc2, zc1), bc, self._ndz, axis=1, tag='X'))

    def _extrude_right(self, obs, idz):
        xc1, zc1 = obs.xz[2], idz[0]
        xc2, zc2 = obs.xz[2], idz[1]
        bc = '{}.{}.'.format(obs.bc[2], self._bc[2])
        for i in range(xc1 + 1, self._nx):
            xc2 = i
            mk = self._xmask[i, zc1+1:zc2]
            if _np.any(mk != 0):
                if _np.all(mk != 0):
                    bc = '{}.W.'.format(obs.bc[2])
                elif _np.any(mk == -2):
                    xc2 = int((xc1+xc2)/2 - 1)
                    bc = '{}.X.'.format(obs.bc[2])
                elif _np.any(mk == 1):
                    xc2 -= 1
                    bc = '{}.X.'.format(obs.bc[1])
                break

        if abs(xc2 - xc1) < self._stencil:
            msg = _exceptions.CloseObstaclesError.msg.format(obs, self._stencil)
            raise _exceptions.CloseObstaclesError(msg)

        self._update_domains(Subdomain((xc1, zc1, xc2, zc2), bc, self._ndx, axis=0, tag='X'))

    def _extrude_left(self, obs, idz):
        """ Extrude left wall of the obstacle. """
        xc1, zc1 = obs.xz[0], idz[0]
        xc2, zc2 = obs.xz[0], idz[1]
        bc = '{}.{}.'.format(self._bc[0], obs.bc[0])
        for i in range(xc1-1, -1, -1):
            xc2 = i
            mk = self._xmask[i, zc1+1:zc2]
            if _np.any(mk != 0):
                if _np.all(mk == -2):
                    bc = 'W.{}.'.format(obs.bc[0])
                elif _np.any(mk == -2):
                    xc2 = int((xc1+xc2)/2 + 1)
                    bc = 'X.{}.'.format(obs.bc[0])
                elif _np.any(mk == 1):
                    xc2 += 1
                    bc = 'X.{}.'.format(obs.bc[1])
                break

        if abs(xc2 - xc1) < self._stencil:
            msg = _exceptions.CloseObstaclesError.msg.format(obs, self._stencil)
            raise _exceptions.CloseObstaclesError(msg)

        self._update_domains(Subdomain((xc2, zc1, xc1, zc2), bc, self._ndx, axis=0, tag='X'))

    def _xbetween(self):
        """ Find domains between obstacles along x axis. """

        idz = ComputationDomains.bounds(self._shape, self._bc, self._obstacles, axis=1)
        if idz:
            void = [(0, i[0], self._nx-1, i[1]) for i in idz if i[2] == 'v']
            void = _utils.remove_dups(void)
            for sub in void:
                self._update_domains(Subdomain(sub,
                                               '{}.{}.'.format(self._bc[0], self._bc[2]),
                                               self._ndx, axis=0, tag='X'))

    def _zbetween(self):
        """ Find domains between obstacles along z axis. """

        idx = ComputationDomains.bounds(self._shape, self._bc, self._obstacles, axis=0)
        if idx:
            void = [(i[0], 0, i[1], self._nz-1) for i in idx if i[2] == 'v']
            void = _utils.remove_dups(void)
            for sub in void:
                self._update_domains(Subdomain(sub,
                                               '.{}.{}'.format(self._bc[1], self._bc[3]),
                                               self._ndz, axis=1, tag='X'))

    def _fill(self):
        """ Search remaining domains along both x and z axises. """

        xmissings = _utils.find_areas(self._xmask, val=0)
        zmissings = _utils.find_areas(self._zmask, val=0)

        for c in xmissings:
            bc = ['X', '.', 'X', '.']
            if c[0] == 0:
                bc[0] = self._bc[0]
            if c[2] == self._nx - 1:
                bc[2] = self._bc[2]
            self._update_domains(Subdomain(c, key=self._ndx, bc=''.join(bc),
                                           axis=0, tag='X'))

        for c in zmissings:
            bc = ['.', 'X', '.', 'X']
            if c[1] == 0:
                bc[1] = self._bc[1]
            if c[3] == self._nz - 1:
                bc[3] = self._bc[3]
            self._update_domains(Subdomain(c, key=self._ndz, bc=''.join(bc),
                                           axis=1, tag='X'))

    def _update_patches(self):
        """ Add patches. TODO : Fix 'else W' if an obstacle has another bc !"""

        for patch in _utils.find_areas(self._xmask, val=-2):
            bc = ['.', '.', '.', '.']
            if patch[0] == 0:
                bc[0] = self._bc[0]
            else:
                bc[0] = 'X' if self._xmask[patch[0]-1, patch[1]] == 1 else 'W'

            if patch[2] == self._nx - 1:
                bc[2] = self._bc[2]
            else:
                bc[2] = 'X' if self._xmask[patch[2]+1, patch[1]] == 1 else 'W'

            self._update_domains(Subdomain(patch, axis=0, key=self._ndx,
                                           bc=''.join(bc), tag='W'))

        for patch in _utils.find_areas(self._zmask, val=-2):
            bc = ['.', '.', '.', '.']
            if patch[1] == 0:
                bc[1] = self._bc[1]
            else:
                bc[1] = 'X' if self._xmask[patch[0], patch[1]-1] == 1 else 'W'

            if patch[3] == self._nz - 1:
                bc[3] = self._bc[3]
            else:
                bc[3] = 'X' if self._xmask[patch[0], patch[3]+1] == 1 else 'W'

            self._update_domains(Subdomain(patch, axis=1, key=self._ndz,
                                           bc=''.join(bc), tag='W'))

    def _update_domains(self, sub, val=1):
        """ Update mask and list of subdomains. """

        if sub.axis == 0:
            self._xmask[sub.sx, sub.sz] = val
            self.dxdomains.append(sub)

        elif sub.axis == 1:
            self._zmask[sub.sx, sub.sz] = val
            self.dzdomains.append(sub)

    @property
    def is_domain_complete(self):
        """ Check if the whole computation domain is subdivided. """
        if self.missing_domains[0].size or self.missing_domains[1].size:
            return False

        return True

    @property
    def is_bc_complete(self):
        """ Check if all boundary conditions are specified. """
        if self.missing_bc:
            return False

        return True

    @property
    def _nd(self):
        """ Index of general subdomain. """
        self._tnd += 1
        return str(self._tnd)

    @property
    def _ndx(self):
        """ Index of general subdomain. """
        self._tndx += 1
        return f'x{self._tndx}'

    @property
    def _ndz(self):
        """ Index of general subdomain. """
        self._tndz += 1
        return f'z{self._tndz}'

    @property
    def missing_domains(self):
        """
        List missings.
        Returns a tuple of ndarray ([array along x], [array along z]]).
        """
        x = _np.argwhere(_np.logical_or(self._xmask == 0, self._xmask == -2))
        z = _np.argwhere(_np.logical_or(self._zmask == 0, self._zmask == -2))
        return x, z

    @property
    def missing_bc(self):
        """ Returns list of subdomain for which at least one bc is missing. """
        return [c for i in range(2) for c in self.listing[i] if c.bc.count('.') > 2]

    @property
    def listing(self):
        """ Returns a tuple containing the domains along x and along z. """
        return self.dxdomains, self.dzdomains

    @property
    def dsdomains(self):
        """ Returns the smallest domain amond derivative domains. """
        return self.dxdomains if len(self.dxdomains) < len(self.dzdomains) else self.dzdomains

    @property
    def fsdomains(self):
        """ Returns the smallest domain among filter domains. """
        return self.fxdomains if len(self.fxdomains) < len(self.fzdomains) else self.fzdomains

    def show_missings(self):
        """ Plot missing subdomains. """
        fig, ax = _plt.subplots(1, 2, figsize=(9, 3))
        ax[0].plot(self.missing_domains[0][:, 0], self.missing_domains[0][:, 1], 'ro')
        ax[0].imshow(self._xmask.T)
        ax[0].axis([-10, self._nx+10, -10, self._nz+10])
        ax[1].plot(self.missing_domains[1][:, 0], self.missing_domains[1][:, 1], 'ro')
        ax[1].imshow(self._zmask.T)
        ax[1].axis([-10, self._nx+10, -10, self._nz+10])
        fig.suptitle(r'Missing points')
        _plt.show()

    @staticmethod
    def bounds(shape, bc, obstacles, axis=0):
        """ Returns a list of tuples representating the bounds of the domain.
        As an example :

        [(0, 10, 'v'), (11, 100, 'o'), (101, 127, 'v')]

        Third element of each tuple represents the type of subdomain (v:void, o:obstacle)

        """


        bounds = []
        mask = ComputationDomains.init_masks(shape, bc, obstacles)[axis] < 0

        for i in range(mask.shape[axis]):
            if axis == 0:
                if mask[i, :].any():
                    mask[i, :] = True
            elif axis == 1:
                if mask[:, i].any():
                    mask[:, i] = True

        for c in _utils.find_areas(mask, val=0):
            bounds.append((c[axis], c[axis+2], 'v'))

        for c in _utils.find_areas(mask, val=1):
            bounds.append((c[axis], c[axis+2], 'o'))

        bounds.sort()

        return bounds

    @staticmethod
    def init_masks(shape, bc, obstacles):
        """ Init masks with obstacles. """

        gdomain = Subdomain([0, 0, shape[0]-1, shape[1]-1], bc=bc)
        xmask = _np.zeros(shape, dtype=int)
        zmask = xmask.copy()

        for obs in obstacles:
            for mask in [xmask, zmask]:
                mask[obs.sx, obs.sz] = -1
                mask[obs.sx, obs.iz] = -2
                mask[obs.ix, obs.sz] = -2

        for obs1, obs2 in _itertools.permutations(obstacles, r=2):
            c = obs1.intersection(obs2)
            if c:
                xmask[c[0], c[1]] = -1
                zmask[c[0], c[1]] = -1

        for obs in obstacles:
            coords = obs.touch(gdomain)
            if coords:
                for c in coords:
                    xmask[c[0], c[1]] = -1
                    zmask[c[0], c[1]] = -1

        _utils.remove_singles(xmask)
        _utils.remove_singles(zmask)

        return xmask, zmask

    def __repr__(self):
        r = '*** x-domains ***\n'
        r += repr(self.dxdomains)
        r += '\n*** z-domains ***\n'
        r += repr(self.dzdomains)
        if len(self.adomains) != 0:
            r += '\n*** PMLs ***\n'
            r += repr(self.adomains)
        return r

    def __str__(self):
        return self.__repr__()


if __name__ == "__main__":

    import templates

    nx, nz = 256, 128

    domain = ComputationDomains((nx, nz),
                                obstacles=templates.helmholtz(nx, nz),
                                stencil=11, bc='WWWW')
    print(domain)
