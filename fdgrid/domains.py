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
#
# pylint: disable=too-many-public-methods
# pylint: disable=too-few-public-methods
# pylint: disable=attribute-defined-outside-init
"""
-----------

Module `domains` provides two classes:

    * Subdomain : dataclass for subdivisions of the computational domain
    * Obstacle : Subdomain subclass with custom boundary possibilities
    * Domain : Container for Subdomain objects

It also provides the fonction `plot_subdomain`.

@author: Cyril Desjouy
"""

__all__ = ['Domain', 'Subdomain', 'Obstacle']

import copy as _copy
import itertools as _itertools
import numpy as _np
import matplotlib.pyplot as _plt
import matplotlib.patches as _patches
import dataclasses as _dataclasses
import fdgrid.utils as _utils


class Domain:
    """ Set of Subdomain objects.

    Parameters
    ----------

    shape : Size of the domain. Must be a 2 elements tuple
    data : list of Subdomain objects or coordinates

    """

    def __init__(self, shape, data=None):

        self.shape = shape
        self.nx, self.nz = shape
        self._data = dict({})
        if data and isinstance(data, (list, tuple)):
            if not isinstance(data[0], Subdomain):
                data = self._subdomain_from_list(data)
            for sub in data:
                if not sub.key:
                    sub.key = self._key
                self._data[sub.key] = sub

        # Additional rigid boundaries when bc is periodic
        self.additional_rigid_bc = None

        # Domain counter
        self._n = 0

    @staticmethod
    def _subdomain_from_list(data):
        subs = []
        for c in data:
            subs.append(Subdomain(c))
        return subs

    def show(self, legend=False, fcolor='b', ecolor='y'):
        """ Represent the Subdomains. """

        _, ax = _plt.subplots(figsize=(9, 9))

        for sub in self:
            rect = _patches.Rectangle((sub.ix[0], sub.iz[0]),
                                      sub.ix[1] - sub.ix[0],
                                      sub.iz[1] - sub.iz[0],
                                      linewidth=3,
                                      edgecolor=ecolor,
                                      facecolor=fcolor,
                                      alpha=0.5)
            ax.add_patch(rect)

        ax.set_xlim(0, self.nx)
        ax.set_ylim(0, self.nz)
        ax.set_aspect('equal')

        major_xticks = range(0, self.nx, 10)
        major_zticks = range(0, self.nz, 10)
        minor_xticks = range(0, self.nx, 2)
        minor_zticks = range(0, self.nz, 2)

        ax.set_xticks(major_xticks)
        ax.set_yticks(major_zticks)
        ax.set_xticks(minor_xticks, minor=True)
        ax.set_yticks(minor_zticks, minor=True)
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

        for sub in self:
            ax.text(*sub.center, f'{sub.key} ({sub.tag})', color='k')

        if legend:
            print(self)

    def keys(self):
        """ Return all keys of Domain instance. """
        return _utils.sort(self._data.keys())

    def sort(self, inplace=False):
        """
        Sort domains.

        If inplace is True, Domain is sorted inplace.
        If inplace is False, return a new instance.
        """

        tmp_list = []
        tmp_dict = {}

        for key in self.keys():
            tmp_list.append(self._data[key])
            tmp_dict[key] = self[key]

        if inplace:
            self._data = tmp_dict
            return None

        return Domain(self.shape, data=tmp_list)

    def copy(self):
        """ Create copy of the instance. """
        return _copy.deepcopy(self)

    def mask(self):
        """ Create mask array of the domain. """

        mask = _np.zeros(self.shape)

        for sub in self:
            mask[sub.sx, sub.sz] += 1

        return mask

    def append(self, val, key=None):
        """
        Append 'val' to the Domain object.

        Arguments :
        ----------
        val : The object to add to the instance.
              Must be a Subdomain object or a list/tuple of Subdomain objects
        key : new key for val
        """

        if isinstance(val, Subdomain):

            if key:
                val.key = str(key)
            elif not val.key and not key:
                val.key = self._key

            if val.key in self._data.keys():
                raise ValueError(f'Domain {val.key} already exists')

            self._data[val.key] = val
            self.sort(inplace=True)

        elif isinstance(val, (tuple, list)):
            for sub in val:
                if isinstance(sub, Subdomain):
                    self.append(sub)
        else:
            raise ValueError('Not a Subdomain object')

    def pop(self, n):
        """ Remove value of index n from the Domain object. """

        self._check_key(str(n))
        self._data.pop(str(n))

    def steal(self, other, key, newkey=None):
        """ Steal key from other. """

        if not newkey:
            newkey = key

        try:
            self.append(other[key], key=newkey)
        except ValueError:
            pass
        finally:
            other.pop(key)

    def split(self, key, index, nbc=None, axis=None):
        """ Split 'key' subdomain.

        Arguments:
            index : int|tuple|list
            nbc : Only if index is a sequence. New bc for the domain defined by 'index'
            axis : 0 or 1. Along which axis to split
        """

        self[key] = self[key].split(index, nbc=nbc, axis=axis)
        self._clean_keys()

    def _clean_keys(self):
        """ Convert 1.1.1 keys to 1.X """

        badkeys = {key.split('.')[0] for key in self.keys() if len(key.split('.')) > 2}
        if badkeys:
            newkeys = self.keys().copy()
            for basekey in badkeys:
                subkey = (i for i in range(1, len(newkeys)+1))
                for idx, key in enumerate(newkeys):
                    if key.split('.')[0] == basekey:
                        newkeys[idx] = '{}.{}'.format(basekey, next(subkey))
            tmp = {}
            for idx, sub in enumerate(self):
                sub.key = newkeys[idx]
                tmp[sub.key] = sub

            self._data = tmp

    def reset_keys(self):
        """ Reset all keys then attribute new ordered keys. """

        tmp = {}
        newkeys = [str(i) for i in range(1, len(self.keys())+1)]
        for key, sub in zip(newkeys, self):
            sub.key = key
            tmp[key] = sub

        self._data = tmp

    def join(self, key1, key2, include_split=False, tag=None):
        """ Join key1 and key2 Subdomain objects. """

        self._check_key(key1)
        self._check_key(key2)

        sub1 = self[key1]
        sub2 = self[key2]

        if not tag:
            tag = sub1.tag

        if not self.isjoinable(sub1, sub2, include_split=include_split):
            raise ValueError("domains can't be joined")

        if sub1.rx == sub2.rx:
            bc = self._join_bc(sub1, sub2, axis=1)
            kwargs = {'bc':bc, 'key':sub1.key, 'axis':sub1.axis, 'tag':tag}

            iz = sub1.iz + sub2.iz
            self[sub1.key] = Subdomain([sub1.ix[0], min(iz), sub1.ix[1], max(iz)], **kwargs)
            self.pop(sub2.key)


        elif sub1.rz == sub2.rz:
            bc = self._join_bc(sub1, sub2, axis=0)
            kwargs = {'bc':bc, 'key':sub1.key, 'axis':sub1.axis, 'tag':tag}

            ix = sub1.ix + sub2.ix
            self[sub1.key] = Subdomain([min(ix), sub1.iz[0], max(ix), sub1.iz[1]], **kwargs)
            self.pop(sub2.key)

    def autojoin(self, include_split=False):
        """ Automatically join domains that can be joined. """

        for sub1, sub2 in _itertools.combinations(self, r=2):
            if self.isjoinable(sub1, sub2, include_split=include_split):
                self.join(sub1.key, sub2.key, include_split=include_split)

    @staticmethod
    def isjoinable(sub1, sub2, include_split=False):
        """ Check if sub1 and sub2 can be joined. """

        if include_split:
            c0 = True
        else:
            c0 = '.' not in sub1.key and '.' not in sub2.key
        c1 = sub1.rz == sub2.rz
        c2 = sub1.ix[1] == sub2.ix[0] - 1 or sub2.ix[1] == sub1.ix[0] - 1
        c3 = sub1.rx == sub2.rx
        c4 = sub1.iz[1] == sub2.iz[0] - 1 or sub2.iz[1] == sub1.iz[0] - 1
        c5 = sub1.axis == sub2.axis

        condition0 = c0 and c5
        condition1 = c1 and c2
        condition2 = c3 and c4

        if (condition0 and condition1) or (condition0 and condition2):
            return True

        return False

    @staticmethod
    def _join_bc(sub1, sub2, axis=None):
        """ Determinae bc for the joined Subdomain. """

        if axis == 0:
            if sub1.axis == 0 and sub1.ix[0] < sub2.ix[0]:
                return f'{sub1.bc[0]}.{sub2.bc[2]}.'

            if sub1.axis == 0 and sub1.ix[0] > sub2.ix[0]:
                return f'{sub2.bc[0]}.{sub1.bc[2]}.'

        if axis == 1:
            if sub1.axis == 1 and sub1.iz[0] < sub2.iz[0]:
                return f'.{sub1.bc[1]}.{sub2.bc[3]}'

            if sub1.axis == 1 and sub1.iz[0] > sub2.iz[0]:
                return f'.{sub2.bc[1]}.{sub1.bc[3]}'

        return sub1.bc

    def periodic_bc(self, mask, axis):
        """ Arange subdomains to take into account periodic boundary condtions. """

        if axis == 0:
            self._xperiod(mask)
            self._get_rigid_bc(self.nx, axis=axis)

        elif axis == 1:
            self._zperiod(mask)
            self._get_rigid_bc(self.nz, axis=axis)

        else:
            raise ValueError('axis must be 0 or 1')

    def _xperiod(self, mask):

        left = _np.argwhere(mask[0, :] != 0).ravel()
        right = _np.argwhere(mask[-1, :] != 0).ravel()

        left_rigid, right_rigid = self._periodic_bounds(left, right)

        for sub in [s for s in self if s.ix[0] == 0]:
            self._periodic_update(sub, left_rigid)

        for sub in [s for s in self if s.ix[1] == self.nx-1]:
            self._periodic_update(sub, right_rigid)

    def _zperiod(self, mask):

        bot = _np.argwhere(mask[:, 0] != 0).ravel()
        top = _np.argwhere(mask[:, -1] != 0).ravel()

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
            bot_rigid = _utils.split_discontinuous(bot_rigid)
        else:
            bot_rigid = []

        if any(top_rigid):
            top_rigid = _utils.split_discontinuous(top_rigid)
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
#                print(f'Process {sub.key} : update bc')
                self[sub.key].bc = nbc

            elif set(range(bound[0], bound[1]+1)).issubset(set(r)):
#                print(f'Process {sub.key} : split subdomain')
                self.split(sub.key, bound, nbc=nbc, axis=axis)

    def _get_rigid_bc(self, N, axis=None):
        """ Returns a list of rigid boundary conditions. """

        self.additional_rigid_bc = []

        for sub in self:
            if sub.xz[axis] == 0 and sub.bc[axis] == 'R':
                if axis == 0:
                    self.additional_rigid_bc.append((0, sub.sz))
                elif axis == 1:
                    self.additional_rigid_bc.append((sub.sx, 0))

            elif sub.xz[axis+2] == N-1 and sub.bc[axis+2] == 'R':
                if axis == 0:
                    self.additional_rigid_bc.append((-1, sub.sz))
                elif axis == 1:
                    self.additional_rigid_bc.append((sub.sx, -1))

    @property
    def _key(self):
        self._n = 1
        while str(self._n) in self._data.keys():
            self._n += 1
        return str(self._n)

    @property
    def xz(self):
        """ Return list of coordinates. """
        if self._data:
            return [s.xz for s in self]

        return None

    @property
    def sx(self):
        """ Return list of slices (over Subdomain x coordinates). """
        if self._data:
            return [s.sx for s in self]

        return None

    @property
    def sz(self):
        """ Return list of slices (over Subdomain z coordinates). """
        if self._data:
            return [s.sz for s in self]

        return None

    @property
    def rx(self):
        """ Return list of ranges (over Subdomain x coordinates). """
        if self._data:
            return [s.rx for s in self]

        return None

    @property
    def rz(self):
        """ Return list of ranges (over Subdomain z coordinates). """
        if self._data:
            return [s.rz for s in self]

        return None

    @property
    def ix(self):
        """ Return list of x coordinates. """
        if self._data:
            return [s.ix for s in self]

        return None

    @property
    def iz(self):
        """ Return list of z coordinates. """
        if self._data:
            return [s.iz for s in self]

        return None

    def _check_key(self, n):
        """ Check if key n exists. """
        if str(n) not in self._data.keys():
            raise KeyError('Sudbomain not found')

    def __setitem__(self, n, val):

        if isinstance(val, (Subdomain, tuple, list)):
            self.pop(str(n))
            self.append(val, n)
        else:
            msg = 'Can only assign Subdomain object or sequence of Subdomain objects'
            raise ValueError(msg)

    def __getitem__(self, n):

        self._check_key(n)

        return self._data[str(n)]

    def __iter__(self):
        return iter(self._data.values())

    def __add__(self, other):

        if not isinstance(other, Domain):
            raise TypeError('Can only concatenate Domain objects')

        if set(self._data.keys()).intersection(set(other.keys())):
            raise ValueError('Duplicate subdomains')

        else:
            tmp = Domain(self.shape)
            for sub in self:
                tmp.append(sub)
            for sub in other:
                tmp.append(sub)

            return tmp

    def __radd__(self, other):
        return self.__add__(other)

    def __len__(self):
        return len(self._data)

    def __str__(self):

        s = ''
        for sub in self:
            s += '\t' + sub.__str__() + '\n'
        return s

    def __repr__(self):
        return self.__str__()


@_dataclasses.dataclass
class Subdomain:
    """ Subdomain of the computation domain.

    Parameters
    ----------

    xz : Coordinates of the Subdomain : left, bottom, right, top
    bc : Boundary conditions. Must be a string of 4 chacracter among 'A', 'R', 'Z' and 'P'
    key : Key of the subdomain. Optional
    axis : 0 or 1. The direction of the subdomain if relevant. Optional.
    tag : str. The type of Subdomain. Optional
    """

    xz: tuple
    bc: str = '....'
    key: str = ''
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
        """ Returns "x" or "z" whether axis is 0 or 1. Return 2 if both. """

        if self.axis == 0:
            return 'x'
        if self.axis == 1:
            return 'z'
        if self.axis == 2:
            return 'xz'
        return None

    @property
    def center(self):
        """ Returns the center of the Subdomain. """
        return [int((self.ix[1]+self.ix[0])/2), int((self.iz[1]+self.iz[0])/2)]

    def intersection(self, other):
        """ Return common border with another object. """

        edges = self._which_edge(other)

        if edges:
            if (0, 2) in edges:
                return self.xz[0], slice(self.xz[1]+1, self.xz[3])

            if (2, 0) in edges:
                return self.xz[2], slice(self.xz[1]+1, self.xz[3])

            if (1, 3) in edges:
                return slice(self.xz[0]+1, self.xz[2]), self.xz[1]

            if (3, 1) in edges:
                return slice(self.xz[0]+1, self.xz[2]), self.xz[3]

        return None

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
            for i, j in _itertools.product([a, b], repeat=2):
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

        subkey = (i for i in range(1, 4))
        xz = []

        if axis is None:
            axis = self.axis

        bounds = self._split_indexes(index, axis)

        for bound in bounds:
            if axis == 0:
                xz.append([self.ix[0], bound[0], self.ix[1], bound[1]])
            elif axis == 1:
                xz.append([bound[0], self.iz[0], bound[1], self.iz[1]])
            else:
                raise ValueError('axis must be 0 or 1')

        bc = self._split_bc(xz, index, nbc, axis)
        tag = self._split_tag(xz)

        return [Subdomain(xz[i], bc[i], key=f'{self.key}.{next(subkey)}',
                          axis=self.axis, tag=tag[i]) for i in range(len(xz))]

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

            if c[1] == index:
                return [(c[0], c[1]), (c[1], c[1])]

            if c[0] <= index < c[1]:
                return [(c[0], index), (index+1, c[1])]

            raise ValueError('index must be in the subdomain')

        elif isinstance(index, (tuple, list)):

            if len(index) == 2:
                index = tuple(sorted(list(index)))
                index_set = set(range(index[0], index[1]+1))
                domain_set = set(r)

                if index_set.issubset(domain_set):
                    remaining = list(domain_set.difference(index_set))
                    lst = _utils.split_discontinuous(remaining)
                    lst.append(index)
                    lst.sort()
                    return lst

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

    def _split_tag(self, xz):
        """ Set right tags to each subdomain. """

        tags = [self.tag]*len(xz)

        for i, c in enumerate(xz):
            if c[0] == c[2] or c[1] == c[3]:
                tags[i] = 'W'

        return tags


@_dataclasses.dataclass
class Edges:
    """ Edges description for moving boundaries. """

    type: str
    f: float
    sx: slice
    sz: slice
    prf: _np.ndarray
    axis: int


class Obstacle(Subdomain):
    """ Obstacle object.

    Parameters
    ----------

    xz : Coordinates of the Subdomain : left, bottom, right, top
    bc : Boundary conditions. Must be a string of 4 chacracter among 'A', 'R', 'Z' and 'P'
    key : Key of the subdomain. Optional
    axis : 0 or 1. The direction of the subdomain if relevant. Optional.
    tag : str. The type of Subdomain. Optional
    """

    def __post_init__(self):
        super().__post_init__()
        self.edges = []

    def set_moving_bc(self, *setup):
        """ Set parameters for moving bc. """

        self.setup = setup

    def _parse_moving_bc(self, bc, setup):

        if bc in ['U', 'V']:
            func = getattr(self, setup['func'], self.flat)
            return [bc], [func], [setup.get('f', 0)], \
                    [setup.get('A', 0)], [setup.get('kwargs', dict())]

        if bc == 'W':
            func = (getattr(self, setup['func'][0], self.flat),
                    getattr(self, setup['func'][1], self.flat))
            return ['U', 'V'], func, setup.get('f', 0), \
                    setup.get('A', 0), setup.get('kwargs', [dict(), dict()])

        raise ValueError("Moving bc must be 'U', 'V', or 'W'")

    def make_moving_bc(self, x, z):
        """ Construct moving boundaries. """

        if hasattr(self, 'setup'):

            self.edges = []
            nbc = 0

            for i, bc in enumerate(self.bc):

                if bc in ['U', 'V', 'W']:

                    comp, func, f, A, kwargs = self._parse_moving_bc(bc, self.setup[nbc])

                    for k, cp in enumerate(comp):

                        if i%2 != 0:
                            lst = [cp, f[k], self.sx, self.xz[i]]
                            lst.append(func[k](x, self.sx, A[k], **kwargs[k]))
                            lst.append(0)
                        else:
                            lst = [cp, f[k], self.xz[i], self.sz]
                            lst.append(func[k](z, self.sz, A[k], **kwargs[k]))
                            lst.append(1)

                        self.edges.append(Edges(*lst))

                    nbc += 1

    @staticmethod
    def sine(c, sc, A, **kwargs):
        """ Sine profile. """

        L = c[sc.stop-1] - c[sc.start]
        n = kwargs.get('n', 1)
        kL = 2*_np.pi/(n*L)

        return A*_np.sin(kL*(c[sc] - c[sc.start]))

    @staticmethod
    def tukey(c, sc, A, **kwargs):
        """ Tapered cosine profile. """

        L = c[sc.stop-1] - c[sc.start]
        N = sc.stop - sc.start

        alpha = kwargs.get('alpha', 0.2)
        edges = L*alpha/2

        Nmin = max(1, abs(c - (c[sc.start] + edges)).argmin() - sc.start)
        Nmax = min(N-1, abs(c - (c[sc.stop] - edges)).argmin() - sc.start)

        nmin = _np.linspace(0, alpha*(N-1)/2, Nmin)
        nmax = _np.linspace((N+1)*(1-alpha/2), N, N - Nmax)

        tukmin = 0.5*(1 + _np.cos(_np.pi*(2*nmin/(alpha*N) + 1)))
        tukmax = 0.5*(1 + _np.cos(_np.pi*(2*nmax/(alpha*N) - 2/alpha + 1)))

        tukmin[0] = 0
        tukmax[-1] = 0

        return  A*_np.concatenate((tukmin, _np.ones(Nmax-Nmin), tukmax))

    @staticmethod
    def flat(*args):
        """ flat profile. """

        return args[2]*_np.ones(args[1].stop-args[1].start)
