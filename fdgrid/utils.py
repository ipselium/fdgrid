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
# Creation Date : 2019-02-13 - 07:48:29
"""
-----------

Some tools for fdgrid.

-----------
"""

import re as _re
import numpy as _np
import scipy.ndimage as _ndi


def sort(l):
    """
    Sort the given iterable in the way that humans expect.
    From https://stackoverflow.com/questions/2669059/how-to-sort-alpha-numeric-set-in-python
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in _re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def is_ordered(lst):
    """ Test if a list is ordered. """
    tmp = lst[:]
    return all(tmp) == all(lst)


def find_areas(array, val=0):
    """
    Find rectangles areas in 'array' with value 'val'. Return a list of
    coordinates as:

        [(xmin, ymin, xmax, ymax), (...)]
    """
    regions, n = _ndi.label(array == val)
    indexes = []
    for i in range(1, n+1):
        c = _np.argwhere(regions == i)
        c_check = remove_dups(split_discontinuous(c[:, 1]))
        if len(c_check) == 1:
            indexes.append(tuple(c[0]) + tuple(c[-1]))
        else:
            idz = remove_dups(split_discontinuous(c[:, 1]))
            idz_size = [i[1]-i[0]+1 for i in idz]
            idx = [(i, list(c[:, 0]).count(i)) for i in set(c[:, 0])]
            idx = [[i[0] for i in idx if i[1] == iz] for iz in idz_size]
            idx = [(ix[0], ix[-1]) for ix in idx]
            coord = [(i[0], j[0], i[1], j[1]) for i, j in zip(idx, idz)]
            for c in coord:
                indexes.append(c)
    return indexes


def remove_singles(array, size=1, oldval=-2, newval=-1):
    """ Replace isolated points of oldval to newval in array.  """
    mask = (array == oldval)
    regions, n = _ndi.label(mask)
    sizes = _np.array(_ndi.sum(mask, regions, range(n + 1)))
    mask = (sizes == size)[regions]
    array[mask] = newval


def split_discontinuous(array):
    """ Split discontinuous array. """
    idx = _np.argwhere(_np.diff(array) != 1).ravel()
    return [(min(l), max(l)) for l in _np.split(array, idx+1)]


def remove_dups(lst):
    """ Remove duplicates from a list. Works with a list of list. """
    tmp = []
    for i in lst:
        if i not in tmp:
            tmp.append(i)
    return tmp


def merge_bc(bc1, bc2):
    """ Merge boundary conditions. """
    bc = ''
    for s1, s2 in zip(bc1, bc2):
        if s1 == '.':
            bc += s2
        elif s2 == '.':
            bc += s1
        else:
            raise ValueError('Boundaries are not compatible')
    return bc


def down_sample(v, N=4):
    """
    Keep 1 point over N.
    N is adjusted to be the nearest multiple of len(v).

    Parameters
    ----------
    v: Object to downsample. Must be ndarray or list object
    N: Keep one point over N

    Returns
    -------
    out: ndarray

    """
    div_list = [i for i in range(2, int((len(v))/2)) if len(v)%i == 0]
    div_min = [abs(i-N) for i in div_list]
    if div_min:
        N = div_list[div_min.index(min(div_min))]
    return _np.array([j for i, j in enumerate(v) if i%N == 0] + [v[-1]])
