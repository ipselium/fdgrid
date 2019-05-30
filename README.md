# FDGrid : Finite difference grid generator


![Grid generator](https://github.com/ipselium/fdgrid/blob/master/docs/fdgrid.png)


## Introducing FDGrid

***FDGrid*** provides some tools to create **regular**, **adaptative** or
**curvilinear** 2D grids. ***FDGrid*** has been designed to work with nsfds2 but can also be used alone. ***FDGrid*** provides tool to:

* Generate 2D meshes : x and z coordinates
* Subdivide the grid into Subdomain objects used by nsfds2

***FDGrid*** provides 3 main types of objects to create meshes:

* `Mesh()`: Create a regular mesh


* `AdaptativeMesh()`: Create a mesh that is adapted around obstacles and close to the boundaries


* `CurvilinearMesh()`: Create a mesh using curvilinear coordinates

***FDGrid*** provides two main objects to create computation domains and/or sets of obstacles:

* `Domain()` : Container for `Subdomain` objects

* `Subdomain()` : Subdivision of the grid


## Requirements and installation

### Requirements

* python > 3.7
* matplotlib > 3.0
* numpy > 1.1
* scipy > 1.1
* ofdlib2 > 0.8

### Installation

Clone the repo and :

```
python setup.py install
```

or install via Pypi :

```
pip install fdgrid
```

## Classical use

### Create set of obstacles

`Domain` and `Subdomain` objects can be used to create sets of obstacles.

* Use `Subdomain` to create an obstacle:

	* First argument is a list of coordinates as *[left, bottom, right, top]*
	* Second argument is the boundary conditions [*(R)igid, (A)bsorbing, (Z)impedance*]

* Use `Domain` to gather all `Subdomain` objects:

	* First argument is the shape of the grid (*tuple*)
	* Keyword argument `data` is a list of `Subdomain` objects

For instance:
```python
from fdgrid import mesh, templates
from fdgrid.domains import Subdomain, Domain

def custom_obstacles(nx, nz):

    geo = [Subdomain([30, 20, 40, 40], 'RRRR'),
           Subdomain([60, 20, 70, 40], 'RRRR'),
           Subdomain([90, 20, 100, 40], 'RRRR')]

    return Domain((nx, nz), data=geo)

nx, nz = 128, 64
dx, dz = 1., 1.
ix0, iz0 = 0, 0
bc = 'ARAR'

mesh2 = mesh.Mesh((nx, nz), (dx, dz), (ix0, iz0), obstacles=custom_obstacles(nx, nz), bc=bc)
mesh2.plot_grid(pml=True)
```

![domains](https://github.com/ipselium/fdgrid/blob/master/docs/regular.png)



### Simple adaptative mesh example

```python
from fdgrid import mesh, templates, domains


shape = (512, 256)	# Dimensions of the grid
steps = (1, 1)		# grid steps
ix0, iz0 = 0, 0		# grid origin
bc = 'RRRR' 		# Boundary conditions : left, bottom, right, top.
			# Can be (R)igid, (A)bsorbing, (P)eriodic, (Z)impedance

# Set up obstacles in the grid with a template
obstacles = templates.testcase1(*shape)

# Generate AdaptativeMesh object
msh = mesh.AdaptativeMesh(shape, steps, (ix0, iz0), obstacles=obstacles, bc=bc)

# Show
msh.plot_grid(axis=True, N=8)
```

![adaptative mesh](https://github.com/ipselium/fdgrid/blob/master/docs/adaptative.png)


### Simple curvilinear mesh example

```python
from fdgrid import mesh, templates
import numpy as np

shape = (256, 256)       # Dimensions of the grid
steps = (1e-4, 1e-4)     # grid steps
origin = (128, 0)        # grid origin
bc = 'RRRR'              # Boundary conditions : left, bottom, right, top.
                         # Can be (R)igid, (A)bsorbing, (P)eriodic, (Z)impedance

# Set up obstacles in the grid with a template
obstacles = templates.helmholtz_double(nx, nz)

# Setup curvilinear transformation
def curv(xn, zn):
    f = 5*dx
    xp = xn.copy()
    zp = zn + np.exp(-np.linspace(0, 10, zn.shape[1]))*np.sin(2*np.pi*f*xn/xn.max()/2)
    return xp, zp

# Generate CurvilinearMesh object
msh = mesh.CurvilinearMesh(shape, steps, origin, obstacles=obstacles, bc=bc, fcurvxz=curv)

# Show physical grid
msh.plot_physical()
```

![curvilinear mesh](https://github.com/ipselium/fdgrid/blob/master/docs/curvilinear.png)

