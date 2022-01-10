Usage
=====

.. _getting_started:

Getting Started
---------------

The most simple way of using the library is to interact with the *BMS* object:

.. code-block:: python
    
   >>> from cideMOD import BMS, DEFAULTS
   >>> bms = BMS('params.json', DEFAULTS.SIMULATION_OPTIONS.value, name="first_run")
   >>> bms.read_test_plan(DEFAULTS.TEST_PLAN.value)
   >>> bms.run_test_plan()

Cell Parameters
----------------


Test Plan
-----------


Finer control
--------------

