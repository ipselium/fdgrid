Introducing fdgrid
==================

|Pypi| |Build| |Licence|


.. image:: https://github.com/ipselium/fdgrid/blob/master/docs/source/images/fdgrid.png


**fdgrid** provides some tools to create **regular**, **adaptative** or
**curvilinear** 2D grids. **fdgrid** has been designed to work with **nsfds2**
but can also be used alone. **fdgrid** provides tool to:

- generate 2D meshes : x and z coordinates,
- subdivide the grid into Subdomain objects used by nsfds2.


Requirements
------------

:python: >= 3.7
:matplotlib: >= 3.0
:numpy: >= 1.1
:scipy: >= 1.1
:ofdlib2: >= 0.8
:mplutils: >= 0.3

Installation
------------

Clone the github repo and

.. code:: console

    $ python setup.py install

or install via Pypi

.. code:: console

    $ pip install fdgrid

Links
-----

- **Documentation:** http://perso.univ-lemans.fr/~cdesjouy/fdgrid/
- **Source code:** https://github.com/ipselium/fdgrid
- **Bug reports:** https://github.com/ipselium/fdgrid/issues
- **nsfds2:** https://github.com/ipselium/nsfds2


.. |Pypi| image:: https://badge.fury.io/py/fdgrid.svg
    :target: https://pypi.org/project/fdgrid
    :alt: Pypi Package

.. |Licence| image:: https://img.shields.io/github/license/ipselium/fdgrid.svg

.. |Build| image:: https://travis-ci.org/ipselium/fdgrid.svg?branch=master
    :target: https://travis-ci.org/ipselium/fdgrid
