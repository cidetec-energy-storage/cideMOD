import dolfinx as dfx
import multiphenicsx.fem
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import numpy as np
import ufl
from solver import NonlinearBlockProblem, NewtonBlockSolver

mesh = dfx.mesh.create_unit_interval(MPI.COMM_WORLD, 100)
boundaries = [(1, lambda x: np.isclose(x[0], 0)),
                (2, lambda x: np.isclose(x[0], 1)),]

left_facets = dfx.mesh.locate_entities(mesh, 0, boundaries[0][1]) 
right_facets = dfx.mesh.locate_entities(mesh, 0, boundaries[1][1])

facet_indices, facet_markers = [], []
fdim = mesh.topology.dim - 1
for (marker, locator) in boundaries:
    facets = dfx.mesh.locate_entities(mesh, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full(len(facets), marker))
facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = dfx.mesh.MeshTags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

PHI_E = dfx.fem.FunctionSpace(mesh, ('CG',1))
LM_IAPP = PHI_E.clone()
phi_e, lm_iapp = dfx.fem.Function(PHI_E), dfx.fem.Function(LM_IAPP)
dphi_e, dlm_iapp = ufl.TrialFunction(PHI_E), ufl.TrialFunction(LM_IAPP)
v_phi, v_lm_iapp = ufl.TestFunction(PHI_E), ufl.TestFunction(LM_IAPP)

dx = ufl.Measure('dx', mesh)
ds = ufl.Measure('ds', mesh, subdomain_data=facet_tag)

I_app = dfx.fem.Constant(mesh, ScalarType(10))
V_app = dfx.fem.Constant(mesh, ScalarType(1))
switch = dfx.fem.Constant(mesh, ScalarType(0))

residuals = [
    ufl.inner(ufl.grad(phi_e),ufl.grad(v_phi)) * dx - lm_iapp*v_phi * ds(2),
    (1-switch)*(I_app-lm_iapp)*v_lm_iapp*ds(2) + switch*(V_app-phi_e)*v_lm_iapp*ds(2)
]
jacobian = [[ufl.derivative(r_i, u_i, du_i) for u_i, du_i in zip([phi_e, lm_iapp],[dphi_e, dlm_iapp])] for r_i in residuals]
bc = dfx.fem.dirichletbc(dfx.fem.Constant(mesh, ScalarType(0)), dfx.fem.locate_dofs_topological(PHI_E, 0, left_facets), PHI_E)

problem = NonlinearBlockProblem(residuals, [phi_e, lm_iapp], [bc], jacobian, restriction=[
    multiphenicsx.fem.DofMapRestriction(PHI_E.dofmap, np.arange(0, PHI_E.dofmap.index_map.size_local+PHI_E.dofmap.index_map.num_ghosts)), 
    multiphenicsx.fem.DofMapRestriction(LM_IAPP.dofmap, dfx.fem.locate_dofs_topological(LM_IAPP, 0, right_facets))])
solver = NewtonBlockSolver(mesh.comm, problem)
solver.solve()

