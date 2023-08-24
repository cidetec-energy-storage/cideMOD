import os

from cideMOD import CSI, ErrorCheck, get_model_options

EVENT1 = {
    "type": "Voltage",  # Voltage, Current, Ah, Wh
    "value": 2,  # Number
    "unit": "V",  #
    "atol": 1e-2,  # Absolute tolerance
    "rtol": 1e-2,  # Relative tolerance
    "goto": "Next",  # Next, End or CV
}

REST = {
    "name": "Storage",
    "type": "Rest",  # 'Current' or 'Voltage' or 'CC' or 'CV' or Rest
    "t_max": {"value": 31, "unit": "day"},  # Maximum duration of step
    "store_delay": -1,  # Store frequency (in timesteps) of internal variables, -1 to deactivate
    "min_step": 10,  # Minimum time step
    "max_step": 3600,  # Maximum time step
    "events": [EVENT1],
}

REST_TEST_PLAN = {
    "initial_state": {
        "SOC": 1,   # State Of Charge (from 0 to 1)
        "exterior_temperature": 298.15  # in K
    },
    "steps": [REST] * 1,
}

case = "Safari_2009"
data_path = "../data/data_{}".format(case)

SIMULATION_OPTIONS = get_model_options(model='P4D', solve_SEI=True, dimensionless=False,
                                       save_path=f"{case}_storage")
cell_data = os.path.join(data_path, "params_cycling.json")

csi = CSI(cell_data, SIMULATION_OPTIONS, REST_TEST_PLAN)
status = csi.run_test_plan()

err = ErrorCheck(csi.problem, status)
