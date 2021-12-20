import os

from cideMOD import BMS, ErrorCheck, ModelOptions

SIMULATION_OPTIONS = ModelOptions(mode='P4D', solve_SEI=True)

EVENT1 = {
    "type": "Voltage",  # Voltage, Current, Ah, Wh
    "value": 2,  # Number
    "unit": "V",  #
    "atol": 1e-2,  # Absolute tolerance
    "rtol": 1e-2,  # Relative tolerance
    "goto": "Next",  # (TODO: Not implemented yet) Next or End
}
EVENT2 = {
    "type": "Voltage",  # Voltage, Current, Ah, Wh
    "value": 4.1,  # Number
    "unit": "V",  #
    "atol": 1e-2,  # Absolute tolerance
    "rtol": 1e-2,  # Relative tolerance
    "goto": "CV",  # (TODO: Not implemented yet) Next or End
}

INPUT1 = {
    "name": "Discharge",
    "type": "Current",  # 'Current' or 'Voltage' or 'CC' or 'CV' or Rest
    "value": -1,  # Must be float, int or string
    "unit": "C",  # One of 'A', 'V', 'mA', 'mV', C
    "t_max": {"value": 60, "unit": "min"},
    "store_delay": 10,
    "min_step": 10,
    # 'adaptive': False,
    "events": [EVENT1],
}
INPUT2 = {
    "name": "Charge",
    "type": "Current",  # 'Current' or 'Voltage' or 'CC' or 'CV' or Rest
    "value": 0.5,  # Must be float, int or string
    "unit": "C",  # One of 'A', 'V', 'mA', 'mV', C
    "t_max": {"value": 120, "unit": "min"},
    "store_delay": 10,
    "min_step": 20,
    # 'adaptive': False,
    "events": [EVENT2],
}

REST = {
    "name": "Storage",
    "type": "Rest",  # 'Current' or 'Voltage' or 'CC' or 'CV' or Rest
    "t_max": {"value": 31, "unit": "day"},
    "store_delay": -1,
    "min_step": 10,
    "max_step": 3600,
    "events": [EVENT1],
}

TEST_PLAN = {
    "initial_state": {"SOC": 1, "exterior_temperature": 298.15},
    "steps": [
        {"name": "Cycling", "type": "Cycle", "count": 700, "steps": [INPUT1, INPUT2]}
    ],
}

REST_TEST_PLAN = {
    "initial_state": {"SOC": 1, "exterior_temperature": 298.15},
    "steps": [REST] * 1,
}

overwrite = False
case = "Safari_2009"
data_path = "../data/data_{}".format(case)

cell_data = os.path.join(data_path, "params.json")


bms = BMS(cell_data, SIMULATION_OPTIONS, name=case)
bms.read_test_plan(TEST_PLAN)
status = bms.run_test_plan()

err = ErrorCheck(bms.problem, status)
