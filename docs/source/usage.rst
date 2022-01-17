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

Specify Model Options
----------------------
The model options can be specified through the :class:`ModelOptions <cideMOD.models.model_options.ModelOptions>` class. For example:

.. code-block:: python
    
   >>> from cideMOD import BMS, DEFAULTS, ModelOptions
   >>> options = ModelOptions(mode='P4D', solve_thermal=True, solve_SEI=True, N_x=30, N_y=20, N_y=20, N_z=20)
   >>> bms = BMS('params.json', options, name="first_run")
   >>> bms.read_test_plan(DEFAULTS.TEST_PLAN.value)
   >>> bms.run_test_plan()

Cell Parameters
----------------

Cell properties and material parameters are specified in a single :code:`json` file or optionally a Python dictionary can be used as well.
Several :ref:`examples <examples>` of different cell datasets are given under the data folder. The :code:`json` file must contain different mandatory objects:

* Structure: 
   The :code:`"structure"` entry determines the configuration of the cell. It is a list containing the tags of the different :ref:`subdomains <mesh>` in order. 
   The easiest configuration would be :code:`["a","s","c"]` corresponding to a cell composed of an anode, separator and cathode. 
   A cell composed of two units could have this other structure :code:`["ncc","a","s","c","pcc","c","s","a","ncc"]`. All the subdomains specified in the "structure" entry must have its own entry defined in the following points.
* Current collectors:
   There is a distinction between the :code:`"positiveCurrentCollector"` and :code:`"negativeCurrentCollector"`, as they often have different properties. These keywords must contain geometry properties and conductivity. If the thermal model is used, thermal properties should be specified as well.
* Electrodes: 
   There is a distinction between the :code:`"positiveElectrode"` and :code:`"negativeElectrode"`, as they have different properties. These keywords must contain information about geometry, porosity, and conductivity. If the thermal model is used, thermal properties should be specified as well.
   Additionally, electrodes should contain a list of the active materials that it contains.
   
   * Active materials:
      Each electrode can have one or more active materials. These are specified as dictionaries with the electrochemical properties that they have.
   
* Separator:
   The :code:`"separator"` keyword must contain geometry and porosity properties. If the thermal model is used, thermal properties should be specified as well.

* Electrolyte:
   The :code:`"electrolyte"` keyword must contain the electrochemical properties such as diffusivity, conductivity, transference number and activity parameter. If the thermal model is used, thermal properties should be specified as well.

In the data folder, there are several examples of such parameter files. This is an example of them:

.. literalinclude:: ../../data/data_Ai_2020/params.json
   :lines: 1,65-

.. note::
   Consider that several parameters as OCP, diffusivities and conductivities can be specified as a function or an array of points. 

Test Plan
-----------
When using the :class:`BMS <cideMOD.bms.battery_system.BMS>` interface to simulate cell behavior, it is necesary to specify a test plan. This can be specified as a json or created programatically as python dictionaries or using cideMOD's :class:`Input <cideMOD.bms.inputs.Input>` and :class:`Trigger <cideMOD.bms.triggers.Trigger>` classes.
An example of a test plan built in the code with the different building blocks is can be found in the :doc:`Use Cases <examples>` section.

Finer control
--------------
If you want to have more control over the inner details of the model, you can directly interact with the :code:`Problem` object, as is the main responsible of assembling the models and running the simulation.
An example of a simple discharge with this approach is given in the :code:`main.py` file under the examples folder:

.. literalinclude:: ../../examples/main.py

Outputs
--------
There are two kind of outputs available. For each simulation, the library saves results to a folder on the fly in case of internal variables, and at the end of the time-loop in the case of global variables.

* Internal variables:
   These are the problem variables, or any derived value that has different values for each point in the cell geometry. These are automatically defined in the code, and written in the `XDMF Format <https://www.xdmf.org/index.php/Main_Page>`_.
   The parameter :code:`store_delay` can be modified to a negative value to supress this output, or to a positive value to specify the frecuency (in timesteps) at which the results are saved to disk.
* Global variables:
   These are overall cell figures (for example cell voltage, current, maximum temperature, etc...), that are calculated and saved as a list of values over time internally in the memory. When the temporal loop finish, they are saved to disk in text files.
