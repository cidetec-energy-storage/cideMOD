from cideMOD import (
    CellParser,
    ErrorCheck,
    NDProblem,
    Problem,
    SolverCrashed,
    Trigger,
    init_results_folder,
    ModelOptions,
)

overwrite = False
case = "Ai_2020"
data_path = "../data/data_{}".format(case)
params = "params.json"

model_options = ModelOptions(clean_on_exit=True)

save_path = init_results_folder(
    case, overwrite=overwrite, copy_files=[f"{data_path}/{params}"]
)
cell = CellParser(params, data_path=data_path)
problem = Problem(cell, model_options, save_path=save_path)
problem.set_cell_state(1, 273 + 25, 273 + 25)
problem.setup()
C_rate = -1
I_app = C_rate * problem.Q
t_f = 3600 /abs(C_rate)*1.25

v_min = Trigger(3, "v")
status = problem.solve_ie(
    min_step=10, i_app=I_app, t_f=t_f, store_delay=10, adaptive=True, triggers=[v_min]
)
err = ErrorCheck(problem, status)

if isinstance(status, SolverCrashed):
    raise status.args[0]
