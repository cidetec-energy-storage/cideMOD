#
# Copyright (c) 2021 CIDETEC Energy Storage.
#
# This file is part of PXD.
#
# PXD is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.#
petsc_options = {
    'log_view': ':snes_profile.txt',
}

snes_options = {
    # 'snes_type': 'newtonls',
    # 'snes_atol': 1e-5,
    # 'snes_rtol': 1e-7,
    'snes_divergence_tolerance': 1e6,
    'snes_line_search_type': 'basic',
    'snes_lag_jacobian': 1,
    # 'snes_monitor_cancel': None,
    # 'snes_converged_reason': None,
}

ksp_options = {
    'ksp_type': 'bcgs',
    # 'ksp_bcgsl_mrpoly': None,
    # 'ksp_view': None,
    'ksp_monitor' : None,
    'ksp_reuse_preconditioner': None,
    # 'ksp_norm_type': 'unpreconditioned',
    # 'ksp_monitor_singular_value': None,
    # 'ksp_converged_reason': None,
    # 'ksp_gmres_restart':  1e2,
}

hypre_options = {
    'pc_type': 'hypre',
    'pc_hypre_boomeramg_cycle_type': 'v',  #Cycle type (choose one of) V W 
    'pc_hypre_boomeramg_max_levels': 25,  #Number of levels (of grids) allowed
    'pc_hypre_boomeramg_max_iter': 1, #Maximum iterations used PER hypre call 
    'pc_hypre_boomeramg_tol': 0. , #Convergence tolerance PER hypre call (0.0 = use a fixed number of iterations) 
    'pc_hypre_boomeramg_numfunctions': 1, #Number of functions 
    'pc_hypre_boomeramg_truncfactor': 0.4 , #Truncation factor for interpolation (0=no truncation)
    'pc_hypre_boomeramg_P_max': 2, #Max elements per row for interpolation operator (0=unlimited) 
    'pc_hypre_boomeramg_agg_nl': 3, #Number of levels of aggressive coarsening 
    'pc_hypre_boomeramg_agg_num_paths': 4, #Number of paths for aggressive coarsening 
    'pc_hypre_boomeramg_strong_threshold': 0.7, # Threshold for being strongly connected (None)
    'pc_hypre_boomeramg_max_row_sum': 0.9, # Maximum row sum
    'pc_hypre_boomeramg_grid_sweeps_all': 1, # Number of sweeps for the up and down grid levels (None)
    'pc_hypre_boomeramg_nodal_coarsen': 0,  #Use a nodal based coarsening 1-6 (HYPRE_BoomerAMGSetNodal)
    'pc_hypre_boomeramg_nodal_coarsen_diag': 0, # Diagonal in strength matrix for nodal based coarsening 0-2 (HYPRE_BoomerAMGSetNodalDiag)
    'pc_hypre_boomeramg_vec_interp_variant': 0, # Variant of algorithm 1-3 (HYPRE_BoomerAMGSetInterpVecVariant)
    'pc_hypre_boomeramg_vec_interp_qmax': 0,  # Max elements per row for each Q (HYPRE_BoomerAMGSetInterpVecQMax)
    'pc_hypre_boomeramg_vec_interp_smooth': False,  #Whether to smooth the interpolation vectors (HYPRE_BoomerAMGSetSmoothInterpVectors)
    'pc_hypre_boomeramg_interp_refine': 0, #Preprocess the interpolation matrix through iterative weight refinement (HYPRE_BoomerAMGSetInterpRefine)
    'pc_hypre_boomeramg_interp_type' : 'ext+i', # Interpolation type (choose one of) classical   direct multipass multipass-wts ext+i ext+i-cc standard standard-wts block block-wtd FF FF1 (None)
    'pc_hypre_boomeramg_grid_sweeps_down': 1, # Number of sweeps for the down cycles (None)
    'pc_hypre_boomeramg_grid_sweeps_up': 1, # Number of sweeps for the up cycles (None)
    'pc_hypre_boomeramg_grid_sweeps_coarse': 1, # Number of sweeps for the coarse level (None)
    'pc_hypre_boomeramg_smooth_num_levels': 25, # Number of levels on which more complex smoothers are used (None)
    'pc_hypre_boomeramg_coarsen_type':  'HMIS', # Coarsen type (choose one of) CLJP Ruge-Stueben  modifiedRuge-Stueben   Falgout  PMIS  HMIS (None)
    # 'pc_hypre_boomeramg_print_statistics': None, # Print statistics (None)
    # 'pc_hypre_boomeramg_print_debug': None #Print debug information (None)
  }

gamg_options = {
    'pc_type': 'ilu',
    # 'hmg_inner_pc_type': 'gamg',
    # 'pc_hmg_reuse_interpolation': True,
}

base_options = {**petsc_options, **snes_options}

def hypre():
    return {**petsc_options, **snes_options, **ksp_options, **hypre_options}

def gamg():
    return {**petsc_options, **snes_options, **ksp_options, **gamg_options}
