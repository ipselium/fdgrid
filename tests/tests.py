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
# Creation Date : 2019-06-28 - 12:14:04
"""
-----------
DOCSTRING

-----------
"""

from fdgrid import Mesh, AdaptativeMesh, CurvilinearMesh
from fdgrid import templates

shp = 256, 128
stps = 1e-4, 1e-4
orgn = 128, 64


for o in templates.test_all():
    for bc in ['AAAA', 'PPPP', 'RRRR', 'PAPA', 'PRPR', 'APAP', 'RPRP']:
        print(f'Cartesian regular with bc {bc} and {o}')
        mesh = Mesh(shp, stps, orgn, obstacles=o(*shp), bc=bc)
        print(f'Cartesian adaptative with {bc} and {o}')
        mesh = AdaptativeMesh(shp, stps, orgn, obstacles=o(*shp), bc=bc)
        print(f'Curvilinear with {bc} and {o}')
        mesh = CurvilinearMesh(shp, stps, orgn, obstacles=o(*shp), bc=bc)
