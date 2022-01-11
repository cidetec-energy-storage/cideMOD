Use cases
==========

In the repository or docker image, several example use cases are included. Additionally several literature datasets are included.

Datasets
---------

- Chen_2020: Graphite-SiliconOxide vs NMC811 cell based on :cite:p:`Chen2020`
- Ai_2020: Graphite vs LCO cell based on :cite:p:`Ai2020`
- Safari_2009: Graphite vs LCO cell based on :cite:p:`Safari2009`

Examples
---------

Single discharge
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This is one of the most basic use cases of cideMOD, to simulate a single discharge using the low-level Problem interface.

.. literalinclude:: ../../examples/main.py


Storage
^^^^^^^^

In this case, using the BMS interface, we can simulate the degradation under rest conditions.

.. literalinclude:: ../../examples/storage.py
    :lines: 1-14,46-56,64-


