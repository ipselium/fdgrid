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

***FDGrid*** provides three main objects to create computation domains and/or sets of obstacles:

* `Domain()` : Container for `Subdomain` objects

* `Subdomain()` : Subdivision of the grid

* `Obstacle()` : Subdomain subclass with moving boundary possibilities

Some geometry templates can be found in the `template` module. Examples are gathered in the following and in `docs`.

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

`Domain` and `Obstacle` objects can be used to create sets of obstacles.

* Use `Obstacle` to create an obstacle:

	* First argument is a list of coordinates as *[left, bottom, right, top]*
	* Second argument is the boundary conditions [*(R)igid, (U) u velocity, (V) v velocity, (W) u & v velocities, (Z)impedance*]

* Use `Domain` to gather all `Obstacle` objects:

	* First argument is the shape of the grid (*tuple*)
	* Keyword argument `data` is a list of `Obstacle` objects

For instance:
```python
from fdgrid import Mesh, Obstacle, Domain

def custom_obstacles(nx, nz):

    geo = [Obstacle([30, 20, 40, 40], 'RRRR'),
           Obstacle([60, 20, 70, 40], 'RRRR'),
           Obstacle([90, 20, 100, 40], 'RRRR')]

    return Domain((nx, nz), data=geo)

nx, nz = 128, 64
dx, dz = 1., 1.
ix0, iz0 = 0, 0
bc = 'ARAR'

mesh = Mesh((nx, nz), (dx, dz), (ix0, iz0), obstacles=custom_obstacles(nx, nz), bc=bc)
mesh.plot_grid(pml=True)
```

![domains](https://github.com/ipselium/fdgrid/blob/master/docs/regular.png)



### Adaptative mesh example

```python
from fdgrid import AdaptativeMesh, templates


shape = (512, 256)	# Dimensions of the grid
steps = (1, 1)		# grid steps
ix0, iz0 = 0, 0		# grid origin
bc = 'RRRR' 		# Boundary conditions : left, bottom, right, top.
			# Can be (R)igid, (A)bsorbing, (P)eriodic, (Z)impedance

# Set up obstacles in the grid with a template
obstacles = templates.testcase1(*shape)

# Generate AdaptativeMesh object
msh = AdaptativeMesh(shape, steps, (ix0, iz0), obstacles=obstacles, bc=bc)

# Show
msh.plot_grid(axis=True, N=8)
```

![adaptative mesh](https://github.com/ipselium/fdgrid/blob/master/docs/adaptative.png)


### Curvilinear mesh example

```python
from fdgrid import CurvilinearMesh, templates
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
msh = CurvilinearMesh(shape, steps, origin, obstacles=obstacles, bc=bc, fcurvxz=curv)

# Show physical grid
msh.plot_physical()
```

![curvilinear mesh](https://github.com/ipselium/fdgrid/blob/master/docs/curvilinear.png)


### Mesh with moving boundaries

```python
from fdgrid import Mesh, Obstacle, Domain

def custom_obstacles(nx, nz, size_percent=20):

    size = int(min(nx, nz)*size_percent/100)

    obs1 = Obstacle([int(nx/2)-size, int(nz/2)-size, int(nx/2)+size, int(nz/2)+size], 'UVRV')
    obs2 = Obstacle([nx-11, 0, nx-1, nz-1], 'URRR')

    obs1.set_moving_bc({'f': 70000, 'A': 1, 'func': 'sine'},
                       {'f': 30000, 'A': -1, 'func': 'tukey'},
                       {'f': 30000, 'A': 1, 'func': 'tukey'})
    obs2.set_moving_bc({'f': 73000, 'A': -1, 'func': 'flat'})

    return Domain((nx, nz), data=[obs1, obs2])


nx, nz = 128, 96
dx, dz = 1., 1.
ix0, iz0 = 0, 0
bc = 'RRRR'

mesh = Mesh((nx, nz), (dx, dz), (ix0, iz0), obstacles=custom_obstacles(nx, nz), bc=bc)
mesh.plot_grid(pml=True, legend=True, bc_profiles=True)
```

![moving boundaries](https://github.com/ipselium/fdgrid/blob/master/docs/moving_bc.png)

