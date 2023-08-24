import os

from cideMOD import CSI, ErrorCheck, get_model_options

EVENT1 = {
    "type": "Voltage",  # Voltage, Current, Ah, Wh
    "value": 2.8,  # Number
    "unit": "V",  #
    "atol": 1e-2,  # Absolute tolerance
    "rtol": 1e-2,  # Relative tolerance
    "goto": "Next",  # Next, End or CV
}
EVENT2 = {
    "type": "Voltage",  # Voltage, Current, Ah, Wh
    "value": 4.1,  # Number
    "unit": "V",  #
    "atol": 1e-2,  # Absolute tolerance
    "rtol": 1e-2,  # Relative tolerance
    "goto": "CV",  # Next, End or CV
}

INPUT1 = {
    "name": "Discharge",
    "type": "Current",  # 'Current' or 'Voltage' or 'CC' or 'CV' or Rest or Cycle
    "value": -1,  # Must be float, int or string
    "unit": "C",  # One of 'A', 'V', 'mA', 'mV', C
    "t_max": {"value": 60, "unit": "min"},  # Maximum duration of step
    "store_delay": 10,  # Store frequency (in timesteps) of internal variables, -1 to deactivate
    "min_step": 10,  # Minimum time step
    # 'adaptive': False, # Wether to adapt timestep or not
    "events": [EVENT1],
}
INPUT2 = {
    "name": "Charge",
    "type": "Current",  # 'Current' or 'Voltage' or 'CC' or 'CV' or Rest or Cycle
    "value": 0.5,  # Must be float, int or string
    "unit": "C",  # One of 'A', 'V', 'mA', 'mV', C
    "t_max": {"value": 120, "unit": "min"},  # Maximum duration of step
    "store_delay": 10,  # Store frequency (in timesteps) of internal variables, -1 to deactivate
    "min_step": 10,  # Minimum time step
    # 'adaptive': False, # Wether to adapt timestep or not
    "events": [EVENT2],
}

CYCLE_INPUT = {
    "name": "Cycling",
    "type": "Cycle",
    "count": 10,  # Number of cycles
    "steps": [INPUT1, INPUT2]  # Steps to be repeated each cycle
}

TEST_PLAN = {
    "initial_state": {
        "SOC": 1,
        "exterior_temperature": 298.15
    },
    "steps": [CYCLE_INPUT],
}

case = "Safari_2009"
data_path = "../data/data_{}".format(case)

SIMULATION_OPTIONS = get_model_options(model='P2D', solve_SEI=True, dimensionless=False,
                                       save_path=f"{case}_cycling")

cell_data = os.path.join(data_path, "params_cycling.json")

csi = CSI(cell_data, SIMULATION_OPTIONS, TEST_PLAN)
status = csi.run_test_plan()

err = ErrorCheck(csi.problem, status)
