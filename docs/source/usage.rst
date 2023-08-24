Usage
=====

Getting Started
---------------

The most simple way of using the library is to interact with the Cell
Simulation Interface, *CSI* object:

.. code-block:: python

   >>> from cideMOD import CSI, DEFAULTS
   >>> csi = CSI('params.json', DEFAULTS.SIMULATION_OPTIONS.value,
                 test_plan = DEFAULTS.TEST_PLAN.value)
   >>> csi.run_test_plan()

Specify Model Options
----------------------
The model options can be specified through the
:meth:`get_model_options <cideMOD.models.model_options.get_model_options>`
method. For example:

.. code-block:: python

   >>> from cideMOD import CSI, DEFAULTS, get_model_options
   >>> options = get_model_options(model='P2D', solve_thermal=False, solve_SEI=True)
   >>> csi = CSI('params.json', options, test_plan=DEFAULTS.TEST_PLAN.value)
   >>> csi.run_test_plan()

Cell Parameters
----------------

Cell properties and material parameters are specified in a single
:code:`json` file or optionally a Python dictionary can be used as
well. Several :ref:`examples <examples>` of different cell datasets are
given under the data folder. The :code:`json` file must contain
different mandatory objects:

* Structure:
   The :code:`"structure"` entry determines the configuration of the
   cell. It is a list containing the tags of the different
   :ref:`subdomains <mesh>` in order. The easiest configuration would
   be :code:`["a","s","c"]` corresponding to a cell composed of an
   anode, separator and cathode. A cell composed of two units could
   have this other structure
   :code:`["ncc","a","s","c","pcc","c","s","a","ncc"]`. All the
   subdomains specified in the "structure" entry must have its own
   entry defined in the following points.

* Current collectors:
   There is a distinction between the :code:`"positive_current_collector"`
   and :code:`"negative_current_ollector"`, as they often have different
   properties. These keywords must contain geometry properties and
   conductivity. If the thermal model is used, thermal properties
   should be specified as well.

* Electrodes:
   There is a distinction between the :code:`"positive_electrode"` and
   :code:`"negative_electrode"`, as they have different properties.
   These keywords must contain information about geometry, porosity,
   and conductivity. If the thermal model is used, thermal properties
   should be specified as well. Additionally, electrodes should contain
   a list of the active materials that it contains.

   * Active materials:
      Each electrode can have one or more active materials. These are
      specified as dictionaries with the electrochemical properties
      that they have.

* Separator:
   The :code:`"separator"` keyword must contain geometry and porosity
   properties. If the thermal model is used, thermal properties should
   be specified as well.

* Electrolyte:
   The :code:`"electrolyte"` keyword must contain the electrochemical
   properties such as diffusivity, conductivity, transference number
   and activity parameter. If the thermal model is used, thermal
   properties should be specified as well.

In the data folder, there are several examples of such parameter files.
This is an example of them:

.. literalinclude:: ../../data/data_Ai_2020/params.json
   :lines: 1,65-

.. note::
   Consider that several parameters as OCP, diffusivities and
   conductivities can be specified as a function or an array of points.

Test Plan
-----------
When using the :class:`CSI <cideMOD.simulation_interface.battery_system.CSI>`
interface to simulate cell behavior, it is necesary to specify a test
plan. This can be specified as a json or created programatically as
python dictionaries or using cideMOD's
:class:`Input <cideMOD.simulation_interface.inputs.Input>`
and :class:`Trigger <cideMOD.numerics.triggers.Trigger>`
classes. An example of a test plan built in the code with the different
building blocks can be found in the :doc:`Use Cases <examples>`
section.

Finer control
--------------
If you want to have more control over the inner details of the model,
you can directly interact with the :code:`Problem` object, as is the
main responsible of assembling the models and running the simulation.
An example of a simple discharge with this approach is given in the
:code:`main.py` file under the examples folder:

.. literalinclude:: ../../examples/main.py

Outputs
--------
There are two kind of outputs available. For each simulation, the
library saves results to a folder on the fly in case of internal
variables, and at the end of the time-loop in the case of global
variables.

* Internal variables:
   These are the problem variables, or any derived value that has
   different values for each point in the cell geometry. These are
   automatically defined in the code, and written in the
   `XDMF Format <https://www.xdmf.org/index.php/Main_Page>`_.
   The parameter :code:`store_delay` can be modified to a negative
   value to suppress this output, or to a positive value to specify the
   frequency (in timesteps) at which the results are saved to the disk.
* Global variables:
   These are overall cell figures (for example; cell voltage, current,
   maximum temperature, etc...), that are calculated and saved as a
   list of values over time internally in the memory. When the time iterations are finished,
   they are saved to disk as text files.
