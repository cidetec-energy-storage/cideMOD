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

Cell properties and material parameters are specified in a single :code:`json` file or optionally a Python dictionary can be used as well.
Several :ref:`examples <examples>` of different battery datasets are given under the data folder. The :code:`json` file must contain different mandatory objects:

* "structure": 
   This entry determines the configuration of the cell. It is a list containing the tags of the different :ref:`subdomains <mesh>` in order. 
   The easiest configuration would be :code:`["a","s","c"]` corresponding to a cell composed of an anode, separator and cathode. 
   A battery composed of two cells could have this other structure :code:`["ncc","a","s","c","pcc","c","s","a","ncc"]`. All the subdomains specified in the "structure" entry must have its own entry defined in the following points.

* "negativeCurrentCollector": 
   asdasdasd
* "positiveCurrentCollector": 
* "negativeElectrode": 
* "positiveElectrode":
* "separator":
* "electrolyte": 

Test Plan
-----------
When using the BMS interface to simulate cell behavior, it is necesary to specify a test plan. This can be specified as a json or created programatically as python dictionaries or using cideMOD's Input, Event and Trigger classes.


Finer control
--------------
If you want to have more control over the inner details of the model, you can directly interact with the :code:`Problem` object, as is the main responsible of assembling the models and running the simulation.
An example of a simple discharge with this approach is given in the :code:`main.py` file under the examples folder:

.. literalinclude:: ../../examples/main.py
