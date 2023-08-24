from cideMOD import (
    CellParser,
    ErrorCheck,
    Problem,
    SolverCrashed,
    Trigger,
    get_model_options,
)

case = "Ai_2020"
data_path = "data/data_{}".format(case)
params = "params.json"

model_options = get_model_options(model='P2D', save_path=f"{case}_discharge", overwrite=False)
cell = CellParser(params, data_path, model_options)
problem = Problem(cell, model_options)
problem.set_cell_state(SoC=1, T_ini=273 + 25, T_ext=273 + 25)
problem.setup()
C_rate = -1
I_app = C_rate * cell.ref_capacity
t_f = 3600 / abs(C_rate) * 1.25

v_min = Trigger(3, "v")
status = problem.solve(
    min_step=10, i_app=I_app, t_f=t_f, store_delay=10, adaptive=False, triggers=[v_min]
)
err = ErrorCheck(problem, status)

if isinstance(status, SolverCrashed):
    raise status.args[0]
