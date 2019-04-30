# FDGrid : Finite difference grid generator


![Grid generator](https://github.com/ipselium/fdgrid/blob/master/docs/fdgrid.png)


## Introducing FDGrid

***FDGrid*** provides some tools to create **regular**, **adaptative** or
**curvilinear** 2D grids. ***FDGrid*** has been designed to work with nsfds2 but can also be used alone. ***FDGrid*** provides tool to :

* Generate 2D meshes : x and z coordinates
* Subdivide the grid into Subdomain objects used by nsfds2

***FDGrid*** provides 3 main types of objects to create meshes :

* `Mesh()` : Create a regular mesh


* `AdaptativeMesh()` : Create a mesh that is adapted around obstacles and close to the boundaries


* `CurvilinearMesh()` : Create a mesh using curvilinear coordinates

***FDGrid*** provides two main objects to creates computation domains and/or sets of obstacles :

* `Domain()` : Container for `Subdomain` objects

* `Subdomain()` : Subdivision of the grid


## Requirement and installation

### Requirements

* python 3.7
* matplotlib
* numpy
* scipy
* ofdlib2

### Installation

```
python setup.py install
```

or

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
def square(nx, nz, size_percent=20):
    """ From fdgrid.templates. """

    size = int(min(nx, nz)*size_percent/100)
    geo = [Subdomain([int(nx/2)-size, int(nz/2)-size,
                      int(nx/2)+size, int(nz/2)+size], 'RRRR')]

    return Domain((nx, nz), data=geo)
```

### Simple adaptative mesh example

```python
from fdgrid import mesh, templates, domains


shape = (512, 256)	# Dimensions of the grid
steps = (1, 1)		# grid steps
ix0, iz0 = 0, 0		# grid origin
bc = 'RRRR' 		# Boundary conditions : left, bottom, right, top.
			# Can be (R)igid, (A)bsorbing, (P)eriodic, (Z)impedance

# Set up obstacles in the grid
obstacles = templates.testcase1(*shape)

# Generate AdaptativeMesh object
mesh1 = mesh.AdaptativeMesh(shape, steps, (ix0, iz0), obstacles=obstacles, bc=bc)

# Show
mesh1.plot_grid(N=8)
```

![adaptative mesh](https://github.com/ipselium/fdgrid/blob/master/docs/adaptative.png)

