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
"""
-----------

Submodule `domain` provides two classes:

    * Subdomain : dataclass for subdivisions of the computational domain
    * Domain : Proceed the subdivision of a computational domain
        * listing : Attribute (tuple of lists) providing the list of subdomains
        (along_x, along_z)
        * missings : Attribute (tuple of ndarrays) providing the list of points
        not included in a subdomain (along_x, along_z)
        * show_missings() : Method that plot a map of the missing points

@author: Cyril Desjouy
"""

import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from .utils import remove_dups, split_discontinuous, get_rectangles


class BoundsNotFoundError(Exception):
    """ Error due to bounds not found. """
    pass


class CloseObstaclesError(Exception):
    """ Error due to obstacles too close. """

    msg = "{} too close to another subdomain for a {} points stencil."


@dataclass
class Subdomain:
    """ Subdomain of the computation domain. """
    xz: list
    bc: str = '....'
    no: str = ''
    axis: str = ''

    def __post_init__(self):

        self.ix = [self.xz[0], self.xz[2]]
        self.iz = [self.xz[1], self.xz[3]]
        self.sx = slice(self.xz[0], self.xz[2]+1)
        self.sz = slice(self.xz[1], self.xz[3]+1)

    def union(self, other):

        edges = self._which_edge(other)

        if edges:
            if (0, 2) in edges:
                return self.xz[0], slice(self.xz[1]+1, self.xz[3])
            elif (2, 0) in edges:
                return self.xz[2], slice(self.xz[1]+1, self.xz[3])
            elif (1, 3) in edges:
                return slice(self.xz[0]+1, self.xz[2]), self.xz[1]
            elif (3, 1) in edges:
                return slice(self.xz[0]+1, self.xz[2]), self.xz[3]
        return

    def _which_edge(self, other):
        """ Search for common edges. """

        edges = []

        if self.axis == 'x' or other.axis == 'x':
            indexes = [(0, 2, 1, 3)]
        elif self.axis == 'z' or other.axis == 'z':
            indexes = [(1, 3, 0, 2)]
        else:
            indexes = [(0, 2, 1, 3), (1, 3, 0, 2)]

        for a, b, c, d in indexes:
            for i, j in itertools.product([a, b], repeat=2):
                c1 = other.xz[j] - 1 <= self.xz[i] <= other.xz[j] + 1
                c2 = self.xz[c] >= other.xz[c] and self.xz[d] <= other.xz[d]
                c3 = other.xz[c] >= self.xz[c] and other.xz[d] <= self.xz[d]
                if c1 and (c2 or c3):
                    edges.append((i, j))

        return edges

    def __matmul__(self, other):
        """ Specify boundary if common edges with "other". """
        edges = self._which_edge(other)
        bc_lst = list(self.bc)

        if not bc_lst:
            bc_lst = 4*['.']

        for edge in edges:
            if other.bc[edge[1]] != '.':
                bc_lst[edge[0]] = other.bc[edge[1]]
            else:
                if other.axis:
                    bc_lst[edge[0]] = 'C'

        self.bc = ''.join(bc_lst)

        return self.bc


@dataclass
class Patch:
    """ Patched area of the computation domain. """
    xz: list
    axis: str = ''

    def __post_init__(self):

        if self.axis == 'x':
            self.ix = [self.xz[0], self.xz[2]]
            self.iz = self.xz[1]

        elif self.axis == 'z':
            self.ix = self.xz[0]
            self.iz = [self.xz[1], self.xz[3]]

        self.sx = slice(self.xz[0], self.xz[2]+1)
        self.sz = slice(self.xz[1], self.xz[3]+1)


class Domain:
    """ Divide computation domain in several subdomains based on obstacle in presence. """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, shape, obstacles, bc, stencil):

        self._stencil = stencil
        self._nx, self._nz = shape
        self._obstacles = obstacles
        self._bc = bc
        self._xmask = np.zeros(shape, dtype=int)
        self._zmask = self._xmask.copy()
        self._xdomains = []
        self._zdomains = []
        self._patches = []
        self._nd = 0                             # Index of the subdomain

        # Fill mask with obstacles
        self._init_masks()

        # Search computation domains
        self._find_domains()

        # Check if domain is completely processed
        self._check_domains()

    def _init_masks(self):
        """ Init masks with obstacles. """

        for obs in self._obstacles:
            for mask in [self._xmask, self._zmask]:
                mask[obs.sx, obs.sz] = -1
                mask[obs.sx, obs.iz] = -2
                mask[obs.ix, obs.sz] = -2

        for obs1, obs2 in itertools.permutations(self._obstacles, r=2):
            c = obs1.union(obs2)
            if c:
                self._xmask[c[0], c[1]] = -1
                self._zmask[c[0], c[1]] = -1

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

    def _check_domains(self):
        """ Make some tests to check if all subdomains are well defined. """
        self._is_domain_complete()
        self._fill_bc()
        self._is_bc_complete()

    def _is_domain_complete(self):
        """ Check if the whole computation domain is subdivided. """
        if self.missings[0].size or self.missings[1].size:
            warnings.warn("Uncomplete computation domain. See domains.show_missings() method.")
            return False
        return True

    def _is_bc_complete(self):
        """ Check if all boundary conditions are specified. """
        lst = [c for i in range(2) for c in self.listing[i] if c.bc.count('.') > 2]
        if lst:
            warnings.warn("Undefined boundary for {}. Check subdomains.".format(lst[0].no))
            return False
        return True

    def _fill_bc(self):
        """ Fill missing boundary conditions. """
        whole_domain = Subdomain([0, 0, self._nx-1, self._nz-1], bc=self._bc, no=0)
        fdomain = self._obstacles + [whole_domain]

        for i in range(2):
            for sub1, sub2 in itertools.permutations(self.listing[i] + fdomain, r=2):
                if sub1.axis or sub2.axis:
                    _ = sub1@sub2

    def _top(self, obs):
        s = self._zmask[obs.sx, obs.iz[1]+1]
        tmp = np.argwhere(s==0).ravel()
        if len(tmp) > 0:
            lst = split_discontinuous(tmp + obs.ix[0])
            for idx in lst:
                self._extrude_top(obs, idx)

    def _bottom(self, obs):
        s = self._zmask[obs.sx, obs.iz[0]-1]
        tmp = np.argwhere(s==0).ravel()
        if len(tmp) > 0:
            lst = split_discontinuous(tmp + obs.ix[0])
            for idx in lst:
                self._extrude_bottom(obs, idx)

    def _right(self, obs):
        s = self._xmask[obs.ix[1]+1, obs.sz]
        tmp = np.argwhere(s==0).ravel()
        if len(tmp) > 0:
            lst = split_discontinuous(tmp + obs.iz[0])
            for idz in lst:
                self._extrude_right(obs, idz)

    def _left(self, obs):
        s = self._xmask[obs.ix[0]-1, obs.sz]
        tmp = np.argwhere(s==0).ravel()
        if len(tmp) > 0:
            lst = split_discontinuous(tmp + obs.iz[0])
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
            if np.any(mk != 0):
                if np.all(mk == -2):
                    bc = '.{}.R'.format(obs.bc[3])
                elif np.any(mk == -2):
                    zc2 = int((zc1+zc2)/2 - 1)
                    bc = '.{}.C'.format(obs.bc[3])
                else:
                    zc2 -= 1
                break

        if abs(zc2 - zc1) < self._stencil:
            raise CloseObstaclesError(CloseObstaclesError.msg.format(obs, self._stencil))

        self._update_domains(Subdomain([xc1, zc1, xc2, zc2], bc, self.nd, 'z'))

    def _extrude_bottom(self, obs, idx):
        """ Extrude bottom wall of the obstacle. """
        xc1, zc1 = idx[0], obs.xz[1]
        xc2, zc2 = idx[1], obs.xz[1]
        bc = '.{}.{}'.format(self._bc[1], obs.bc[1])
        for j in range(zc1-1, -1, -1):
            zc2 = j
            mk = self._zmask[xc1+1:xc2, j]
            if np.any(mk != 0):
                if np.all(mk == -2):
                    bc = '.R.{}'.format(obs.bc[1])
                elif np.any(mk == -2):
                    zc2 = int((zc1+zc2)/2 + 1)
                    bc = '.C.{}'.format(obs.bc[1])
                else:
                    zc2 +=1
                break

        if abs(zc2 - zc1) < self._stencil:
            raise CloseObstaclesError(CloseObstaclesError.msg.format(obs, self._stencil))

        self._update_domains(Subdomain([xc1, zc2, xc2, zc1], bc, self.nd, 'z'))

    def _extrude_right(self, obs, idz):
        xc1, zc1 = obs.xz[2], idz[0]
        xc2, zc2 = obs.xz[2], idz[1]
        bc = '{}.{}.'.format(obs.bc[2], self._bc[2])
        for i in range(xc1 + 1, self._nx):
            xc2 = i
            mk = self._xmask[i, zc1+1:zc2]
            if np.any(mk != 0):
                if np.all(mk != 0):
                    bc = '{}.R.'.format(obs.bc[2])
                elif np.any(mk == -2):
                    xc2 = int((xc1+xc2)/2 - 1)
                    bc = '{}.C.'.format(obs.bc[2])
                else:
                    xc2 -= 1
                break

        if abs(xc2 - xc1) < self._stencil:
            print(obs, abs(xc2-xc1))
            raise CloseObstaclesError(CloseObstaclesError.msg.format(obs, self._stencil))

        self._update_domains(Subdomain([xc1, zc1, xc2, zc2], bc, self.nd, 'x'))

    def _extrude_left(self, obs, idz):
        """ Extrude left wall of the obstacle. """
        xc1, zc1 = obs.xz[0], idz[0]
        xc2, zc2 = obs.xz[0], idz[1]
        bc = '{}.{}.'.format(self._bc[0], obs.bc[0])
        for i in range(xc1-1, -1, -1):
            xc2 = i
            mk = self._xmask[i, zc1+1:zc2]
            if np.any(mk != 0):
                if np.all(mk == -2):
                    bc = 'R.{}.'.format(obs.bc[0])
                elif np.any(mk == -2):
                    xc2 = int((xc1+xc2)/2 + 1)
                    bc = 'C.{}.'.format(obs.bc[0])
                else:
                    xc2 += 1
                break

        if abs(xc2 - xc1) < self._stencil:
            raise CloseObstaclesError(CloseObstaclesError.msg.format(obs, self._stencil))

        self._update_domains(Subdomain([xc2, zc1, xc1, zc2], bc, self.nd, 'x'))

    def _xbetween(self):
        """ Find domains between obstacles along z axis. """

        idz = Domain.bounds(self._nz, self._obstacles, axis='z')
        if idz:
            void = [[0, i[0], self._nx-1, i[1]] for i in idz if i[2] == 'v']
            void = remove_dups(void)
            for sub in void:
                self._update_domains(Subdomain(sub, '{}.{}.'.format(self._bc[0], self._bc[2]),
                                               self.nd, 'x'))

    def _zbetween(self):
        """ Find domains between obstacles along z axis. """

        idx = Domain.bounds(self._nx, self._obstacles, axis='x')
        if idx:
            void = [[i[0], 0, i[1], self._nz-1] for i in idx if i[2] == 'v']
            void = remove_dups(void)
            for sub in void:
                self._update_domains(Subdomain(sub, '.{}.{}'.format(self._bc[1], self._bc[3]),
                                               self.nd, 'z'))

    def _fill(self):
        """ Search remaining domains along both x and z axises. """
        xmissings = get_rectangles(self._xmask)
        zmissings = get_rectangles(self._zmask)
        for c in xmissings:
            self._update_domains(Subdomain(c, no=self.nd, axis='x'))

        for c in zmissings:
            self._update_domains(Subdomain(c, no=self.nd, axis='z'))

    def _update_patches(self, val=1):
        """ Add patches. """

        xmiss, zmiss = self.missings

        idx = list(set(xmiss[:, 1]))
        idz = list(set(zmiss[:, 0]))

        lx = np.array([[xmiss[i, 0] for i in range(len(xmiss[:, 0])) if xmiss[i, 1] == ix] for ix in idx])
        lz = np.array([[zmiss[i, 1] for i in range(len(zmiss[:, 0])) if zmiss[i, 0] == iz] for iz in idz])

#        print(lx, lz)
#
#        for c in range(len(idx)):
#            self._patches.append(Patch([lx[c].min(), idx[c], lx[c].max(), idx[c]], axis='x'))
#            self._xmask[self._patches[-1].sx, self._patches[-1].sz] = val
#
#        for c in range(len(idz)):
#            self._patches.append(Patch([idz[c], lz[c].min(), idz[c], lz[c].max()], axis='z'))
#            self._zmask[self._patches[-1].sx, self._patches[-1].sz] = val

    def _update_domains(self, sub, val=1):
        """ Update mask and list of subdomains. """

        if sub.axis == 'x':
            self._xmask[sub.sx, sub.sz] = val
            self._xdomains.append(sub)

        elif sub.axis == 'z':
            self._zmask[sub.sx, sub.sz] = val
            self._zdomains.append(sub)

    @property
    def nd(self):
        """ Index of each subdomain. """
        self._nd += 1
        return self._nd

    def show_missings(self):
        """ Plot missing subdomains. """
        fig, ax = plt.subplots(1, 2, figsize=(9, 3))
        ax[0].plot(self.missings[0][:, 0], self.missings[0][:, 1], 'ro')
        ax[0].imshow(self._xmask.T)
        ax[0].axis([-10, self._nx+10, -10, self._nz+10])
        ax[1].plot(self.missings[1][:, 0], self.missings[1][:, 1], 'ro')
        ax[1].imshow(self._zmask.T)
        ax[1].axis([-10, self._nx+10, -10, self._nz+10])
        fig.suptitle(r'Missing points')
        plt.show()

    @property
    def missings(self):
        """ List missings. Returns a tuple of ndarray ([array along x], [array along z]]). """
        x = np.argwhere(np.logical_or(self._xmask == 0, self._xmask == -2))
        z = np.argwhere(np.logical_or(self._zmask == 0, self._zmask == -2))
        return x, z

    @property
    def listing(self):
        """ Returns a tuple containing the domains along x and along z. """
        return self._xdomains, self._zdomains

    def __repr__(self):
        r = '*** x-domains ***\n'
        for s in self._xdomains:
            r += '\t{}\n'.format(s)
        r += '\n*** z-domains ***\n'
        for s in self._zdomains:
            r += '\t{}\n'.format(s)
        return r

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def bounds(N, obstacles, axis='x'):
        """ Returns a list of tuples representating the bounds of each subdomain.
        As an example :

        [(0, 10, 'v'), (10, 100, 'o'), (90, 127, 'o')]

        Third element of each tuple represents the type of subdomain (v:void, o:obstacle)

        """

        if axis == 'x':
            i1, i2 = 0, 2
        elif axis == 'z':
            i1, i2 = 1, 3

        obs_bounds = [(o.xz[i1], o.xz[i2], 'o') for o in obstacles]
        obs_sets = [set(range(o[0], o[1])) for o in obs_bounds]

        void_bounds = []
        void_sets = list(set(range(N)).difference(*obs_sets))
        void_sets.sort()

        if void_sets:
            void_bounds = [(void_sets[i], void_sets[i+1])
                           for i in list(*np.where(np.diff(void_sets) != 1))]
            void_bounds = [min(void_sets), *[i for sub in void_bounds for i in sub], max(void_sets)]
            void_bounds = [(void_bounds[i]+1, void_bounds[i+1], 'v') if void_bounds[i] !=0
                            else (void_bounds[i], void_bounds[i+1], 'v')
                           for i in range(0, len(void_bounds)-1, 2)
                           if void_bounds[i] != void_bounds[i+1]]


        out = obs_bounds + void_bounds
        out.sort()

        return out

    @staticmethod
    def split_bounds(lst, N):
        """ Split bounds in order to get sets that don't have common ranges.
        As an example :

        lst = [(0, 10, 'v'), (10, 180, 'o'), (90, 120), 'o']
        out = [(0, 10, 'v'), (10, 90, 'o'), (90, 120, 'o'), (120, 180, 'o')]
        """

        out = lst[:]

        maxit = 0
        while not all([out[i][1] == out[i+1][0] for i in range(len(out)-1)]):

            bound_sets = [set(range(s[0], s[1])) for s in out]

            for i, j in itertools.combinations(range(len(bound_sets)), 2):
                if bound_sets[i].intersection(bound_sets[j]):
                    if out[j][1] <= out[i][1]:
                        out.append((out[j][1], out[i][1], out[i][2]))
                        out[i] = (out[i][0], out[j][0], out[i][2])
                    elif out[j][1] > out[i][1]:
                        out[j] = (out[i][1], out[j][1], out[i][2])
                    out.sort()
                    break

            if maxit > len(lst)**2:
                raise BoundsNotFoundError("Bound algorithm dit not converge")

            maxit += 1

        if out[0][0] != 0:
            out.append((0, out[0][0], 'v'))

        if out[-1][1] != N-1:
            out.append((out[-1][1], N-1, 'v'))

        out.sort()

        return [i for i in out if i[0] != i[1]]


if __name__ == "__main__":

    import templates

    nx, nz = 256, 128

    domain = Domain((nx, nz), obstacles=templates.helmholtz(nx, nz), bc='RRRR')
    print(domain)
