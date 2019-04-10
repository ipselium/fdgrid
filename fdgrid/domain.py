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


import re
import copy
import string
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from .utils import remove_dups, remove_singles, is_ordered
from .utils import split_discontinuous, find_areas, sort


class CloseObstaclesError(Exception):
    """ Error due to obstacles too close. """

    msg = "{} too close to another subdomain for a {} points stencil."


class Domains:

    def __init__(self, shape, data=None):

        self.shape = shape
        self.nx, self.nz = shape
        self._data = dict({})
        if isinstance(data, list):
            for sub in data:
                if not sub.no:
                    sub.no = self._no
                self._data[sub.no] = sub

    def keys(self):
        """ Return all keys of Domains instance. """
        return sort(self._data.keys())

    def sort(self, inplace=False):
        """ 
        Sort domains.

        If inplace is True, Domains is sorted inplace. 
        If inplace is False, return a new instance. 
        """

        tmp_list = []
        tmp_dict = {}

        for key in self.keys():
            tmp_list.append(self._data[key])
            tmp_dict[key] = self[key]

        if inplace:
            self._data = tmp_dict
            return
        else:
            return Domains(self.shape, data=tmp_list)

    def copy(self):
        """ Create copy of the instance. """
        return copy.deepcopy(self)

    def mask(self):
        """ Create mask array of the domain. """

        mask = np.zeros(self.shape)

        for sub in self:
            mask[sub.sx, sub.sz] += 1

        return mask

    def append(self, val, key=None):
        """ 
        Append 'val' to the Domains object. 
        
        Arguments :
        ----------
        val : The object to add to the instance.
              Must be a Subdomain object or a list/tuple of Subdomain objects
        key : Only considered is Subdomain.no is empty
        """

        if isinstance(val, Subdomain):

            if val.no in self._data.keys():
                raise ValueError(f'Domain {val.no} already exists')

            if key and not val.no:
                val.no = str(key)
            elif not val.no and not key:
                val.no = self._no

            self._data[val.no] = val

        elif isinstance(val, tuple) or isinstance(val, list):
            for sub in val:
                if isinstance(sub, Subdomain):
                    self.append(sub)
        else:
            raise ValueError('Not a Subdomain object')

    def pop(self, n):
        """ Remove value of index n from the Domains object. """

        self._check_key(str(n))
        self._data.pop(str(n))

    def split(self, key, index, nbc=None, axis=None):
        """ Split 'key' subdomain. 
        
        Arguments:
            index : int|tuple|list
            nbc : Only if index is a sequence. New bc for the domain defined by 'index'
            axis : 0 or 1. Along which axis to split
        """

        self[key] = self[key].split(index, nbc=nbc, axis=axis)

    def join(self, key1, key2, include_split=False, tag=None):

        self._check_key(key1)
        self._check_key(key2)

        sub1 = self[key1]
        sub2 = self[key2]

        if not tag:
            tag = sub1.tag

        kwargs = {'bc':sub1.bc, 'no':sub1.no, 'axis':sub1.axis, 'tag':tag}

        if not self._is_joinable(sub1, sub2, include_split=include_split):
            raise ValueError("domains can't be joined")

        if sub1.rx == sub2.rx:
            iz = sub1.iz + sub2.iz
            self[sub1.no] = Subdomain([sub1.ix[0], min(iz), sub1.ix[1], max(iz)], **kwargs)
            self.pop(sub2.no)


        elif sub1.rz == sub2.rz:
            ix = sub1.ix + sub2.ix
            self[sub1.no] = Subdomain([min(ix), sub1.iz[0], max(ix), sub1.iz[1]], **kwargs)
            self.pop(sub2.no)

    def autojoin(self, include_split=False):
        """ Automatically join domains that can be joined. """

        for sub1, sub2 in itertools.combinations(self, r=2):
            if self._is_joinable(sub1, sub2, include_split=include_split):
                self.join(sub1.no, sub2.no, include_split=include_split)

    @staticmethod
    def _is_joinable(sub1, sub2, include_split=False):
        """ Check if sub1 and sub2 can be joined. """

        if include_split:
            condition0 = True
        else:
            condition0 = sub1.no.isnumeric() and sub2.no.isnumeric()
        condition1 = sub1.rz == sub2.rz
        condition2 = sub1.ix[1] == sub2.ix[0] - 1 or sub2.ix[1] == sub1.ix[0] - 1
        condition3 = sub1.rx == sub2.rx
        condition4 = sub1.iz[1] == sub2.iz[0] - 1 or sub2.iz[1] == sub1.iz[0] - 1
        condition5 = sub1.axis == sub2.axis

        if condition0 and condition1 and condition2 and condition5:
            return True
        elif condition0 and condition3 and condition4 and condition5:
            return True
        else:
            return False

    def pml(self, axis):
        pass

    def periodic_bc(self, mask, axis):
        """ Arange subdomains to take into account periodic boundary condtions. """

        if axis == 0:
            self._xperiod(mask)
            self.rigid_bc = self.get_rigid_bc(self.nx, axis=axis)

        elif axis == 1:
            self._zperiod(mask)
            self.rigid_bc = self.get_rigid_bc(self.nz, axis=axis)

        else:
            raise ValueError('axis must be 0 or 1')

    def _xperiod(self, mask):

        left = np.argwhere(mask[0,:] != 0).ravel()
        right = np.argwhere(mask[-1,:] != 0).ravel()

        left_rigid, right_rigid = self._periodic_bounds(left, right)

        for sub in [s for s in self if s.ix[0] == 0]:
            self._periodic_update(sub, left_rigid)

        for sub in [s for s in self if s.ix[1] == self.nx-1]:
            self._periodic_update(sub, right_rigid)

    def _zperiod(self, mask):

        bot = np.argwhere(mask[:,0] != 0).ravel()
        top = np.argwhere(mask[:,-1] != 0).ravel()

        bot_rigid, top_rigid = self._periodic_bounds(bot, top)

        for sub in [s for s in self if s.iz[0] == 0]:
            self._periodic_update(sub, bot_rigid)

        for sub in [s for s in self if s.iz[1] == self.nz-1]:
            self._periodic_update(sub, top_rigid)

    @staticmethod
    def _periodic_bounds(bot, top):

        bot_rigid = list(set(top).difference(set(bot)))
        top_rigid = list(set(bot).difference(set(top)))

        if any(bot_rigid):
            bot_rigid = split_discontinuous(bot_rigid)
        else:
            bot_rigid = []

        if any(top_rigid):
            top_rigid = split_discontinuous(top_rigid)
        else:
            top_rigid = []

        return bot_rigid, top_rigid

    def _periodic_update(self, sub, bounds):

        if sub.axis == 0:
            r = getattr(sub, 'rz')
            axis = 0
        elif sub.axis == 1:
            r = getattr(sub, 'rx')
            axis = 1

        nbc = sub.bc.replace('P', 'R')

        for bound in bounds:
            if set(range(bound[0], bound[1]+1)).issuperset(set(r)):
#                print(f'Process {sub.no} : update bc')
                self[sub.no].bc = nbc

            elif set(range(bound[0], bound[1]+1)).issubset(set(r)):
#                print(f'Process {sub.no} : split subdomain')
                self.split(sub.no, bound, nbc=nbc, axis=axis)

    def get_rigid_bc(self, N, axis=None):
        """ Returns a list of rigid boundary conditions. """

        rigid_bc = []

        for sub in self:
            if sub.xz[axis] == 0 and sub.bc[axis] == 'R':
                if axis == 0:
                    rigid_bc.append((0, sub.sz))
                elif axis == 1:
                    rigid_bc.append((sub.sx, 0))

            elif sub.xz[axis+2] == N-1 and sub.bc[axis+2] == 'R':
                if axis == 0:
                    rigid_bc.append((-1, sub.sz))
                elif axis == 1:
                    rigid_bc.append((sub.sx, -1))

        return rigid_bc

    @property
    def _no(self):
        self._n = 1
        while str(self._n) in self._data.keys():
            self._n += 1
        return str(self._n)

    @property
    def xz(self):
        """ Return list of coordinates. """
        if self._data:
            return [s.xz for s in self]
        else:
            return

    @property
    def sx(self):
        """ Return list of slices (over Subdomain x coordinates). """
        if self._data:
            return [s.sx for s in self]
        else:
            return

    @property
    def sz(self):
        """ Return list of slices (over Subdomain z coordinates). """
        if self._data:
            return [s.sz for s in self]
        else:
            return

    @property
    def rx(self):
        """ Return list of ranges (over Subdomain x coordinates). """
        if self._data:
            return [s.rx for s in self]
        else:
            return

    @property
    def rz(self):
        """ Return list of ranges (over Subdomain z coordinates). """
        if self._data:
            return [s.rz for s in self]
        else:
            return

    @property
    def ix(self):
        """ Return list of x coordinates. """
        if self._data:
            return [s.ix for s in self]
        else:
            return

    @property
    def iz(self):
        """ Return list of z coordinates. """
        if self._data:
            return [s.iz for s in self]
        else:
            return

    def _check_key(self, n):
        """ Check if key n exists. """
        if str(n) not in self._data.keys():
            raise KeyError('Sudbomain not found')

    def __setitem__(self, n, val):

        if isinstance(val, Subdomain) or isinstance(val, tuple) or isinstance(val, list):
            self.pop(str(n))
            self.append(val, n)
        else:
            raise ValueError('Can only assign Subdomain object or sequence of Subdomain objects')

    def __getitem__(self, n):

        self._check_key(n)

        return self._data[str(n)]

    def __iter__(self):
        return iter(self._data.values())

    def __add__(self, other):

        if not isinstance(other, Domains):
            raise TypeError('Can only concatenate Domains objects')

        elif set(self._data.keys()).intersection(set(other._data.keys())):
            raise ValueError('Duplicate subdomains')

        else:
            tmp = Domains(self.shape)
            for sub in self:
                tmp.append(sub)
            for sub in other:
                tmp.append(sub)

            return tmp

    def __radd__(self, other):
        return __add__(self, other)

    def __len__(self):
        return len(self._data)

    def __str__(self):

        s = ''
        for sub in self:
            s += '\t' + sub.__str__() + '\n'
        return s

    def __repr__(self):
        return self.__str__()


@dataclass
class Subdomain:
    """ Subdomain of the computation domain. """
    xz: tuple
    bc: str = '....'
    no: str = ''
    axis: int = -1
    tag: str = ''

    def __post_init__(self):

        self.ix = (self.xz[0], self.xz[2])
        self.iz = (self.xz[1], self.xz[3])
        self.rx = range(self.xz[0], self.xz[2]+1)
        self.rz = range(self.xz[1], self.xz[3]+1)
        self.sx = slice(self.xz[0], self.xz[2]+1)
        self.sz = slice(self.xz[1], self.xz[3]+1)

    @property
    def axname(self):
        return "x" if self.axis == 0 else "z" if self.axis == 1 else None

    def intersection(self, other):
        """ Return common borders with another object. """

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

    def touch(self, other):
        """ Return borders that touch the domain. """

        edges = self._which_edge(other)
        coords = []

        if edges:
            for c in edges:
                if c == (0, 0):
                    coords.append((self.xz[0], slice(self.xz[1]+1, self.xz[3])))
                elif c == (1, 1):
                    coords.append((slice(self.xz[0]+1, self.xz[2]), self.xz[1]))
                elif c == (2, 2):
                    coords.append((self.xz[2], slice(self.xz[1]+1, self.xz[3])))
                elif c == (3, 3):
                    coords.append((slice(self.xz[0]+1, self.xz[2]), self.xz[3]))
        return coords

    def _which_edge(self, other):
        """ Search for common edges. """

        edges = []

        if self.axis == 0 or other.axis == 0:
            indexes = [(0, 2, 1, 3)]
        elif self.axis == 1 or other.axis == 1:
            indexes = [(1, 3, 0, 2)]
        else:
            indexes = [(0, 2, 1, 3), (1, 3, 0, 2)]

        for a, b, c, d in indexes:
            for i, j in itertools.product([a, b], repeat=2):
                c1 = other.xz[j] - 1 <= self.xz[i] <= other.xz[j] + 1
                c2 = self.xz[c] >= other.xz[c] and self.xz[d] <= other.xz[d]
                if c1 and c2:
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
                    bc_lst[edge[0]] = 'X'

        self.bc = ''.join(bc_lst)

        return self.bc

    def split(self, index, nbc=None, axis=None):
        """ Split the subdomain along axis'.
        If index is integer, split into two Subdomain objects at 'index'.
        If index is tuple/list (len(index) must be 2), split into 3 (or less)
        Subdomain objects.

        Arguments:
            index : int|tuple|list
            nbc : Only if index is a sequence. New bc for the domain defined by 'index'
            axis : 0 or 1. Along which axis to split
        """

        letters = string.ascii_lowercase
        xz = []

        if axis is None:
            axis = self.axis

        bounds = self._split_indexes(index, axis)

        for bound in bounds:
            if axis==0:
                xz.append([self.ix[0], bound[0], self.ix[1], bound[1]])
            elif axis==1:
                xz.append([bound[0], self.iz[0], bound[1], self.iz[1]])
            else:
                raise ValueError('axis must be 0 or 1')

        bc = self._split_bc(xz, index, nbc, axis)

        return [Subdomain(xz[i], bc[i], no=f'{self.no}{letters[i]}',
                axis=self.axis, tag=self.tag) for i in range(len(xz))]

    def _split_indexes(self, index, axis):

        if axis == 0:
            r = getattr(self, 'rz')
            c = getattr(self, 'iz')
        elif axis == 1:
            r = getattr(self, 'rx')
            c = getattr(self, 'ix')
        else:
            raise ValueError('axis must be 0 or 1')

        if isinstance(index, int):

            if c[0] < index < c[1]:
                return [(c[0], index), (index+1, c[1])]

            else:
                raise ValueError('index must be in the subdomain')

        elif isinstance(index, tuple) or isinstance(index, list):

            if len(index) == 2:
                index = tuple(sorted(list(index)))
                index_set = set(range(index[0], index[1]+1))
                domain_set = set(r)

                if index_set.issubset(domain_set):
                    remaining = list(domain_set.difference(index_set))
                    lst = split_discontinuous(remaining)
                    lst.append(index)
                    lst.sort()
                    return lst
                else:
                    raise ValueError('indexes must be in the domain')
            else:
                raise ValueError('index must be of length 2')
        else:
            raise ValueError('index must be int, list, or tuple')

    def _split_bc(self, xz, index, nbc, axis):
        """ Determine bc when splitting Subdomain. """

        if axis == 0:
            cid = 1
        elif axis == 1:
            cid = 0

        if nbc and len(xz) > 1:
            bc = [self.bc]*len(xz)
            for idx, coord in enumerate(xz):
                if set(range(coord[cid], coord[cid+2])).issubset(set(range(*index))):
                    bc = [self.bc]*len(xz)
                    bc[idx] = nbc

        elif axis != self.axis and self.axis == 0:
            bc = ['X.X.']*len(xz)
            bc[0] = '{}.X.'.format(self.bc[0])
            bc[-1] = 'X.{}.'.format(self.bc[2])

        elif axis != self.axis and self.axis == 1:
            bc = ['.X.X']*len(xz)
            bc[0] = '.{}.X'.format(self.bc[1])
            bc[-1] = '.X.{}'.format(self.bc[3])

        else:
            bc = [self.bc]*len(xz)

        return bc


class CoDomain:
    """ Divide computation domain in several subdomains based on obstacle in presence. """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, shape, obstacles, bc, stencil, autojoin=True):

        self._stencil = stencil
        self._shape = shape
        self._nx, self._nz = shape
        self._obstacles = obstacles
        self._bc = bc
        self._xdomains = Domains(shape)
        self._zdomains = Domains(shape)
        self._nd = 0                             # Index of the subdomains

        # Initialize masks
        self._xmask, self._zmask = self.init_masks(shape, bc, obstacles)

        # Search computation domains
        self._find_domains()

        # Check if domain is completely processed
        self._check_subdomains()

        # Update Domains considering periodic bc
        if re.match(r'P.P.', self._bc):
            mask, _ = self.init_masks(shape, bc, obstacles)
            self._xdomains.periodic_bc(mask, axis=0)
        if re.match(r'.P.P', self._bc):
            mask, _ = self.init_masks(shape, bc, obstacles)
            self._zdomains.periodic_bc(mask, axis=1)

        # Update Domains considering PML
        if self._bc[0] is 'A':
            self._xdomains.pml(axis=0)
        if self._bc[1] is 'A':
            self._zdomains.pml(axis=1)
        if self._bc[2] is 'A':
            self._xdomains.pml(axis=0)
        if self._bc[3] is 'A':
            self._zdomains.pml(axis=1)

        # join Subdomains that can be joined
        if autojoin:
            self._xdomains.autojoin()
            self._zdomains.autojoin()

        self._xdomains.sort(inplace=True)
        self._zdomains.sort(inplace=True)

    @staticmethod
    def init_masks(shape, bc, obstacles):
        """ Init masks with obstacles. """

        whole_domain = Subdomain([0, 0, shape[0]-1, shape[1]-1], bc=bc, no=0, tag='a')
        xmask = np.zeros(shape, dtype=int)
        zmask = xmask.copy()

        for obs in obstacles:
            for mask in [xmask, zmask]:
                mask[obs.sx, obs.sz] = -1
                mask[obs.sx, obs.iz] = -2
                mask[obs.ix, obs.sz] = -2

        for obs1, obs2 in itertools.permutations(obstacles, r=2):
            c = obs1.intersection(obs2)
            if c:
                xmask[c[0], c[1]] = -1
                zmask[c[0], c[1]] = -1

        for obs in obstacles:
            coords = obs.touch(whole_domain)
            if coords:
                for c in coords:
                    xmask[c[0], c[1]] = -1
                    zmask[c[0], c[1]] = -1

        remove_singles(xmask)
        remove_singles(zmask)

        return xmask, zmask

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
            warnings.warn("Uncomplete computation domain. See domains.show_missings() method.")
        if not self.is_bc_complete:
            self._fill_bc()

    def _fill_bc(self):
        """ Fill missing boundary conditions. """

        # The whole domain
        whole_domain = Subdomain([0, 0, self._nx-1, self._nz-1], bc=self._bc, no=0, tag='a')

        fdomain = self._obstacles + [whole_domain]

#        for i in range(2):
#            listing = [s for s in self.listing[i] if not s.tag]
#            for sub1, sub2 in itertools.permutations(listing + fdomain, r=2):
#                if sub1.axis or sub2.axis:
#                    _ = sub1@sub2

        if not self.is_bc_complete:
            msg = "Undefined boundary for {}. Check subdomains."
            warnings.warn(msg.format(self.missing_bc[0].no))

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
                    bc = '.{}.X'.format(obs.bc[3])
                elif np.any(mk == 1):
                    zc2 -= 1
                    bc = '.{}.X'.format(obs.bc[3])
                break

        if abs(zc2 - zc1) < self._stencil:
            raise CloseObstaclesError(CloseObstaclesError.msg.format(obs, self._stencil))

        self._update_domains(Subdomain((xc1, zc1, xc2, zc2), bc, self.nd, axis=1, tag='X'))

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
                    bc = '.X.{}'.format(obs.bc[1])
                elif np.any(mk == 1):
                    zc2 += 1
                    bc = '.X.{}'.format(obs.bc[1])
                break

        if abs(zc2 - zc1) < self._stencil:
            raise CloseObstaclesError(CloseObstaclesError.msg.format(obs, self._stencil))

        self._update_domains(Subdomain((xc1, zc2, xc2, zc1), bc, self.nd, axis=1, tag='X'))

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
                    bc = '{}.X.'.format(obs.bc[2])
                elif np.any(mk == 1):
                    xc2 -= 1
                    bc = '{}.X.'.format(obs.bc[1])
                break

        if abs(xc2 - xc1) < self._stencil:
            raise CloseObstaclesError(CloseObstaclesError.msg.format(obs, self._stencil))

        self._update_domains(Subdomain((xc1, zc1, xc2, zc2), bc, self.nd, axis=0, tag='X'))

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
                    bc = 'X.{}.'.format(obs.bc[0])
                elif np.any(mk == 1):
                    xc2 += 1
                    bc = 'X.{}.'.format(obs.bc[1])
                break

        if abs(xc2 - xc1) < self._stencil:
            raise CloseObstaclesError(CloseObstaclesError.msg.format(obs, self._stencil))

        self._update_domains(Subdomain((xc2, zc1, xc1, zc2), bc, self.nd, axis=0, tag='X'))

    def _xbetween(self):
        """ Find domains between obstacles along x axis. """

        idz = CoDomain.bounds(self._shape, self._bc, self._obstacles, axis=1)
        if idz:
            void = [(0, i[0], self._nx-1, i[1]) for i in idz if i[2] == 'v']
            void = remove_dups(void)
            for sub in void:
                self._update_domains(Subdomain(sub,
                                               '{}.{}.'.format(self._bc[0], self._bc[2]),
                                               self.nd, axis=0, tag='X'))

    def _zbetween(self):
        """ Find domains between obstacles along z axis. """

        idx = CoDomain.bounds(self._shape, self._bc, self._obstacles, axis=0)
        if idx:
            void = [(i[0], 0, i[1], self._nz-1) for i in idx if i[2] == 'v']
            void = remove_dups(void)
            for sub in void:
                self._update_domains(Subdomain(sub,
                                               '.{}.{}'.format(self._bc[1], self._bc[3]),
                                               self.nd, axis=1, tag='X'))

    def _fill(self):
        """ Search remaining domains along both x and z axises. """
        xmissings = find_areas(self._xmask, val=0)
        zmissings = find_areas(self._zmask, val=0)
        for c in xmissings:

            bc = ['X', '.', 'X', '.']
            if c[0] == 0:
                bc[0] = self._bc[0]
            if c[2] == self._nx - 1:
                bc[2] = self._bc[2]

            self._update_domains(Subdomain(c, no=self.nd, bc=''.join(bc), axis=0, tag='X'))

        for c in zmissings:

            bc = ['.', 'X', '.', 'X']
            if c[1] == 0:
                bc[1] = self._bc[1]
            if c[3] == self._nz - 1:
                bc[3] == self._bc[3]

            self._update_domains(Subdomain(c, no=self.nd, bc=''.join(bc), axis=1, tag='X'))

    def _update_patches(self, val=1):
        """ Add patches. TODO : Fix 'else R' if an obstacle has another bc !"""

        for patch in find_areas(self._xmask, val=-2):
            bc = ['.', '.', '.', '.']
            if patch[0] == 0:
                bc[0] = self._bc[0]
            else:
                bc[0] = 'X' if self._xmask[patch[0]-1, patch[1]] == 1 else 'R'

            if patch[2] == self._nx - 1:
                bc[2] = self._bc[2]
            else:
                bc[2] = 'X' if self._xmask[patch[2]+1, patch[1]] == 1 else 'R'

            self._update_domains(Subdomain(patch, axis=0, no=self.nd,
                                           bc=''.join(bc), tag='W'))

        for patch in find_areas(self._zmask, val=-2):
            bc = ['.', '.', '.', '.']
            if patch[1] == 0:
                bc[1] = self._bc[1]
            else:
                bc[1] = 'X' if self._xmask[patch[0], patch[1]-1] == 1 else 'R'

            if patch[3] == self._nz - 1:
                bc[3] = self._bc[3]
            else:
                bc[3] = 'X' if self._xmask[patch[0], patch[3]+1] == 1 else 'R'

            self._update_domains(Subdomain(patch, axis=1, no=self.nd,
                                           bc=''.join(bc), tag='W'))

    def _update_domains(self, sub, val=1):
        """ Update mask and list of subdomains. """

        if sub.axis == 0:
            self._xmask[sub.sx, sub.sz] = val
            self._xdomains.append(sub)

        elif sub.axis == 1:
            self._zmask[sub.sx, sub.sz] = val
            self._zdomains.append(sub)

    @property
    def is_domain_complete(self):
        """ Check if the whole computation domain is subdivided. """
        if self.missing_domains[0].size or self.missing_domains[1].size:
            return False
        else:
            return True

    @property
    def is_bc_complete(self):
        """ Check if all boundary conditions are specified. """
        if self.missing_bc:
            return False
        else:
            return True

    @property
    def nd(self):
        """ Index of each subdomain. """
        self._nd += 1
        return str(self._nd)

    @property
    def missing_domains(self):
        """
        List missings.
        Returns a tuple of ndarray ([array along x], [array along z]]).
        """
        x = np.argwhere(np.logical_or(self._xmask == 0, self._xmask == -2))
        z = np.argwhere(np.logical_or(self._zmask == 0, self._zmask == -2))
        return x, z

    @property
    def missing_bc(self):
        """ Returns list of subdomain for which at least one bc is missing. """
        return [c for i in range(2) for c in self.listing[i] if c.bc.count('.') > 2]

    @property
    def listing(self):
        """ Returns a tuple containing the domains along x and along z. """
        return self._xdomains, self._zdomains

    def __repr__(self):
        r = '*** x-domains ***\n'
        r += repr(self._xdomains)
        r += '\n*** z-domains ***\n'
        r += repr(self._zdomains)
        return r

    def __str__(self):
        return self.__repr__()

    def show_missings(self):
        """ Plot missing subdomains. """
        fig, ax = plt.subplots(1, 2, figsize=(9, 3))
        ax[0].plot(self.missing_domains[0][:, 0], self.missing_domains[0][:, 1], 'ro')
        ax[0].imshow(self._xmask.T)
        ax[0].axis([-10, self._nx+10, -10, self._nz+10])
        ax[1].plot(self.missing_domains[1][:, 0], self.missing_domains[1][:, 1], 'ro')
        ax[1].imshow(self._zmask.T)
        ax[1].axis([-10, self._nx+10, -10, self._nz+10])
        fig.suptitle(r'Missing points')
        plt.show()

    @staticmethod
    def bounds(shape, bc, obstacles, axis=0):
        """ Returns a list of tuples representating the bounds of the domain.
        As an example :

        [(0, 10, 'v'), (11, 100, 'o'), (101, 127, 'v')]

        Third element of each tuple represents the type of subdomain (v:void, o:obstacle)

        """


        bounds = []
        mask = CoDomain.init_masks(shape, bc, obstacles)[axis] < 0

        for i in range(mask.shape[axis]):
            if axis == 0:
                if mask[i, :].any() == True:
                    mask[i, :] = True
            elif axis == 1:
                if mask[:, i].any() == True:
                    mask[:, i] = True

        for c in find_areas(mask, val=0):
            bounds.append((c[axis], c[axis+2], 'v'))

        for c in find_areas(mask, val=1):
            bounds.append((c[axis], c[axis+2], 'o'))

        bounds.sort()

        return bounds


if __name__ == "__main__":

    import templates

    nx, nz = 256, 128

    domain = CoDomain((nx, nz), obstacles=templates.helmholtz(nx, nz), bc='RRRR')
    print(domain)
