Introduction
=============

.. _FEniCS: https://fenicsproject.org/download/archive/
.. _multiphenics: https://github.com/multiphenics/multiphenics
.. _gmsh: https://gmsh.info/

CIDETEC's open-source tool **cideMOD** is based on the Doyle-Fuller-Newman 
model :cite:p:`Doyle1993` in which the physicochemical equations
are solved by Finite Element methods using the `FEniCS`_
:cite:p:`fenics` library. It enables doing physics-based battery cell
(herein after cell) simulations with a wide variety of use cases, from
different drive cycles to studies of the SEI growth under storage
conditions.

**cideMOD** is a pseudo X-dimensional (pXD) model that extends the
original pseudo 2-dimensional (P2D) model, proposed by Newman and
co-workers, from 1D to 2D and 3D cell geometries. Therefore, charge
balance, mass balance and reaction kinetics, as well as energy balance,
are spatially resolved for the entire cell geometry, considering the
inhomogeneity of cell state properties.

Check out the :doc:`Installation <install>` section for instructions to
install the software or the :doc:`Usage <usage>` section for
information on how to use the library.

.. rubric:: Requirements

- `FEniCS`_ :cite:p:`fenics`
- `multiphenics`_ :cite:p:`multiphenics`
- `gmsh`_ :cite:p:`gmsh`

.. rubric:: Authors

- Clara Ganuza (cganuza@cidetec.es)
- Ruben Parra (rparra@cidetec.es)
- Diego del Olmo (ddelolmo@cidetec.es)
- Raul Ciria (rciria@cidetec.es)
- Elixabete Ayerbe (eayerbe@cidetec.es)

.. rubric:: License

cideMOD is copyright (C) 2022 of CIDETEC Energy Storage and is
distributed under the terms of the Affero GNU General Public License
(GPL) version 3 or later.
