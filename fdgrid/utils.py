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

Some tools.

@author: Cyril Desjouy
"""

import numpy as np


def get_rectangles(mask):
    """ find rectangles in a ndarray of 0 and 1. """

    output = []

    for ix in range(mask.shape[0]):
        for iz in range(mask.shape[1]):
            if mask[ix, iz] == 0:
                _get_rectangle_coordinates(ix, iz, mask, output)

    return output


def _get_rectangle_coordinates(ix, iz, mask, output):
    """ Subfunction of get_rectangles. """

    output.append([ix, iz])

    for i in range(ix, mask.shape[0]):
        if i == mask.shape[0]-1:
            output[-1].append(i)
        elif mask[i][iz] != 0 :
            output[-1].append(i-1)
            break


    for j in range(iz, mask.shape[1]):
        if j == mask.shape[1]-1:
            output[-1].append(j)
        if np.any(mask[output[-1][0]:output[-1][2], j]) != 0:
            output[-1].append(j-1)
            break

    mask[output[-1][0]:output[-1][2]+1, output[-1][1]:output[-1][3]+1] = 1




def split_discontinuous(array):
    idx = np.argwhere(np.diff(array) != 1).ravel()
    return [[min(l), max(l)] for l in np.split(array, idx+1)]


def remove_dups(lst):
    """ Remove duplicates from a list. Works with a list of list. """
    tmp = []
    for i in lst:
        if i not in tmp:
            tmp.append(i)
    return tmp


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
    return np.array([j for i, j in enumerate(v) if i%N == 0] + [v[-1]])
