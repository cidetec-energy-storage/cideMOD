Use cases
==========

In the repository or docker image, several use cases are
included together with corresponding literature datasets used in the examples.

.. _examples:

Datasets
---------

- Chen_2020: Graphite-SiliconOxide vs NMC811 cell based on :cite:p:`Chen2020`
    Electrochemical parameters
- Ai_2020: Graphite vs LCO cell based on :cite:p:`Ai2020`
    Electrochemical and thermal parameters
- Safari_2009: Graphite vs LCO cell based on :cite:p:`Safari2009`
    Electrochemical and SEI parameters

Examples
---------

Single discharge
^^^^^^^^^^^^^^^^^^
This is one of the most basic use cases of cideMOD, to simulate a
single discharge using the low-level Problem interface.

.. literalinclude:: ../../examples/main.py

Storage
^^^^^^^^

In this case, using the CSI interface, we can simulate the
degradation under rest conditions.

.. literalinclude:: ../../examples/storage.py

Cycling
^^^^^^^^

Using a different test plan, we can simulate a cycling protocol:

.. literalinclude:: ../../examples/cycling.py

