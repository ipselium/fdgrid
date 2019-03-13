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
from .utils import remove_dups


def _find_end(i, j, mask, output, index, include_boundaries):
    """ From https://www.geeksforgeeks.org/find-rectangles-filled-0/ """

    # pylint: disable=too-many-arguments

    flagc = 0     # flag to check column edge case
    flagr = 0     # flag to check row edge case

    for m in range(i, mask.shape[0]):

        if mask[m][j] == 1:
            flagr = 1
            break          # breaks where first 1 encounters

        if mask[m][j] == 5:
            continue           # pass because already processed

        for n in range(j, mask.shape[1]):

            if mask[m][n] != 0:
                flagc = 1
                break      # breaks where first 1 encounters

            mask[m][n] = 5    # fill rectangle elements with any  number

    if flagr == 1 and not include_boundaries:
        output[index].append(m-1)
    else:
        output[index].append(m)  # If end point touch the boundary

    if flagc == 1 and not include_boundaries:
        output[index].append(n-1)
    else:
        output[index].append(n)  # If end point touch the boundary


def get_rectangle_coordinates(array, include_boundaries=False):
    """
    From https://www.geeksforgeeks.org/find-rectangles-filled-0/

    Get coordinates of rectangle areas of 0 in an array

    Parameters
    ----------
    array:

    Returns
    -------
    output: list of rectangle coordinates [[xmin, ymin, xmax, ymax], [...]]

    """

    mask = array.copy()
    output = []
    index = -1

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 0:
                if include_boundaries:
                    output.append([max(0, i-1), max(0, j-1)])  # Initial position
                else:
                    output.append([i, j])  # Initial position
                index = index + 1      # Used for last postion
                _find_end(i, j, mask, output, index, include_boundaries)

    return output


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

    def _which_edge(self, other):
        """ Search for common edges. """

        edges = []

        if self.axis == 'x':
            indexes = [(0, 2, 1, 3)]
        elif self.axis == 'z':
            indexes = [(1, 3, 0, 2)]
        else:
            indexes = [(0, 2, 1, 3), (1, 3, 0, 2)]

        for a, b, c, d in indexes:
            for i, j in itertools.product([a, b], repeat=2):
                if self.xz[i] == other.xz[j] and self.xz[c] >= other.xz[c] and self.xz[d] <= other.xz[d]:
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

        self.bc = ''.join(bc_lst)

        return self.bc


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
        self._nd = 0                             # Index of the subdomain

        # Fill mask with obstacles
        for obs in self._obstacles:
            self._xmask[obs.xz[0]:obs.xz[2]+1, obs.xz[1]:obs.xz[3]+1] = 1
            self._zmask[obs.xz[0]:obs.xz[2]+1, obs.xz[1]:obs.xz[3]+1] = 1

        # Search computation domains
        for obs in self._obstacles:
            self._right(obs)
            self._left(obs)
            self._top(obs)
            self._bottom(obs)

        self._xbetween()
        self._zbetween()
        self._fill()
        self._fill_bc()
        self._check_domains()

    def _check_domains(self):
        """ Make some tests to check if all subdomains are well defined. """
        self._check_missings()
        self._check_bc()

    def _check_missings(self):
        """ Check if the whole computation domain is subdivided. """
        if self.missings[0].size or self.missings[1].size:
            warnings.warn("Uncomplete computation domain. See domains.show_missings() method.")

    def _check_bc(self):
        """ Check if all boundary conditions are specified. """
        lst = [c for i in range(2) for c in self.listing[i] if c.bc.count('.') > 2]
        if lst:
            warnings.warn("Undefined boundary for {}. Check subdomains.".format(lst[0].no))

    def _fill_bc(self):
        """ Fill missing boundary conditions. """
        all_subdomains = [c for i in range(2) for c in self.listing[i]]
        empty_subdomain = [s for s in all_subdomains if s.bc == '....']
        whole_domain = Subdomain([0, 0, self._nx-1, self._nz-1], bc=self._bc)

        for subdomain in empty_subdomain:

            _ = subdomain@whole_domain

            for other in all_subdomains:
                _ = subdomain@other

            for obstacle in self._obstacles:
                _ = subdomain@obstacle

    def _top(self, obs):
        """ Extrude top wall of the obstacle. """
        xc1, zc1 = obs.xz[0], obs.xz[3]
        xc2, zc2 = obs.xz[2], obs.xz[3]
        bc = '.R.{}'.format(self._bc[3])
        for j in range(zc1+1, self._nz):
            zc2 = j
            if np.any(self._zmask[xc1+1:xc2, j] != 0):
                if np.all(self._zmask[xc1+1:xc2, j] != 0):
                    bc = '.R.R'
                else:
                    zc2 = int((zc1+zc2)/2)
                    if 2 <= zc2 - zc1 < self._stencil:
                        raise CloseObstaclesError(CloseObstaclesError.msg.format(obs, self._stencil))
                    bc = '.R.C'
                break

        if zc1 + 1 < zc2:
            self._update_domains(Subdomain([xc1, zc1, xc2, zc2], bc, self.nd, 'z'))

    def _bottom(self, obs):
        """ Extrude bottom wall of the obstacle. """
        xc1, zc1 = obs.xz[0], obs.xz[1]
        xc2, zc2 = obs.xz[2], obs.xz[1]
        bc = '.{}.R'.format(self._bc[1])
        for j in range(zc1-1, -1, -1):
            zc2 = j
            if np.any(self._zmask[xc1+1:xc2, j] != 0):
                if np.all(self._zmask[xc1+1:xc2, j] != 0):
                    bc = '.R.R'
                else:
                    bc = '.C.R'
                break

        if zc1 - 1 > zc2:
            self._update_domains(Subdomain([xc1, zc2, xc2, zc1], bc, self.nd, 'z'))

    def _right(self, obs):
        """ Extrude right wall of the obstacle. """
        xc1, zc1 = obs.xz[2], obs.xz[1]
        xc2, zc2 = obs.xz[2], obs.xz[3]
        bc = 'R.{}.'.format(self._bc[2])
        for i in range(xc1 + 1, self._nx):
            xc2 = i
            if np.any(self._xmask[i, zc1+1:zc2] != 0):
                if np.all(self._xmask[i, zc1+1:zc2] != 0):
                    bc = 'R.R.'
                else:
                    xc2 = int((xc1+xc2)/2)
                    if 2 <= xc2 - xc1 < self._stencil:
                        raise CloseObstaclesError(CloseObstaclesError.msg.format(obs, self._stencil))
                    bc = 'R.C.'
                break

        if xc1 + 1 < xc2:
            self._update_domains(Subdomain([xc1, zc1, xc2, zc2], bc, self.nd, 'x'))

    def _left(self, obs):
        """ Extrude left wall of the obstacle. """
        xc1, zc1 = obs.xz[0], obs.xz[1]
        xc2, zc2 = obs.xz[0], obs.xz[3]
        bc = '{}.R.'.format(self._bc[0])
        for i in range(xc1-1, -1, -1):
            xc2 = i
            if np.any(self._xmask[i, zc1+1:zc2] != 0):
                if np.all(self._xmask[i, zc1+1:zc2] != 0):
                    bc = 'R.R.'
                else:
                    bc = 'C.R.'
                break

        if xc1 - 1 > xc2:
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
                self._update_domains(Subdomain(sub, '{}.{}.'.format(self._bc[1], self._bc[3]),
                                               self.nd, 'z'))

    def _fill(self):
        """ Search remaining domains along both x and z axises. """
        xmissings = get_rectangle_coordinates(self._xmask, include_boundaries=True)
        zmissings = get_rectangle_coordinates(self._zmask, include_boundaries=True)
        for c in xmissings:
            self._update_domains(Subdomain(c, no=self.nd, axis='x'))

        for c in zmissings:
            self._update_domains(Subdomain(c, no=self.nd, axis='z'))

    def _update_domains(self, c, val=1):
        """ Update mask and list of subdomains. """
        if c.xz[2] == self._nx - 1:
            xslice = slice(c.xz[0], None)
        else:
            xslice = slice(c.xz[0], c.xz[2]+1)

        if c.xz[3] == self._nz - 1:
            zslice = slice(c.xz[1], None)
        else:
            zslice = slice(c.xz[1], c.xz[3]+1)

        if c.axis == 'x':
            self._xmask[xslice, zslice] = val
            self._xdomains.append(c)

        elif c.axis == 'z':
            self._zmask[xslice, zslice] = val
            self._zdomains.append(c)

    @property
    def nd(self):
        """ Index of each subdomain. """
        self._nd += 1
        return self._nd

    def show_missings(self):
        """ Plot missing subdomains. """
        plt.figure(figsize=(9, 3))
        plt.subplot(1, 2, 1)
        plt.plot(self.missings[0][:, 0], self.missings[0][:, 1], 'o')
        plt.axis([-10, self._nx+10, -10, self._nz+10])
        plt.subplot(1, 2, 2)
        plt.plot(self.missings[1][:, 0], self.missings[1][:, 1], 'o')
        plt.axis([-10, self._nx+10, -10, self._nz+10])
        plt.show()

    @property
    def missings(self):
        """ List missings. Returns a tuple of ndarray ([array along x], [array along z]]). """
        return np.argwhere(self._xmask == 0), np.argwhere(self._zmask == 0)

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
            void_bounds = [(void_bounds[i], void_bounds[i+1]+1, 'v') if void_bounds[i+1] != N-1
                           else (void_bounds[i], void_bounds[i+1], 'v')
                           for i in range(0, len(void_bounds)-1, 2)
                           if void_bounds[i] != void_bounds[i+1]]

        out = obs_bounds + void_bounds
        out.sort()

        return out

    @staticmethod
    def split_bounds(lst, N):
        """ Split bound in order to get sets that don't have common ranges.
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
