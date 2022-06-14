import dolfinx as dfx
import multiphenicsx.fem
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import numpy as np
import ufl
from solver import NonlinearBlockProblem, NewtonBlockSolver

n= 30
mesh = dfx.mesh.create_unit_square(MPI.COMM_WORLD, n, 100*n)
cdim = mesh.topology.dim
fdim = mesh.topology.dim - 1

# Define boundaries
boundaries = [(1, lambda x: np.isclose(x[0], 0)),
                (2, lambda x: np.isclose(x[0], 1)),]
left_facets = dfx.mesh.locate_entities(mesh, fdim, boundaries[0][1]) 
right_facets = dfx.mesh.locate_entities(mesh, fdim, boundaries[1][1])
facet_indices, facet_markers = [], []
for (marker, locator) in boundaries:
    facets = dfx.mesh.locate_entities(mesh, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full(len(facets), marker))
facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = dfx.mesh.MeshTags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

# Define domains
domains = [
    (1, lambda x: x[0]<=0.2),
    (2, lambda x: x[0]>=0.2),
]
left_cells = dfx.mesh.locate_entities(mesh, cdim, domains[0][1]) 
right_cells = dfx.mesh.locate_entities(mesh, cdim, domains[1][1])
cell_indices, cell_markers = [], []
for (marker, locator) in domains:
    cells = dfx.mesh.locate_entities(mesh, cdim, locator)
    cell_indices.append(cells)
    cell_markers.append(np.full(len(cells), marker))
cell_indices = np.array(np.hstack(cell_indices), dtype=np.int32)
cell_markers = np.array(np.hstack(cell_markers), dtype=np.int32)
sorted_cells = np.argsort(cell_indices)
cell_tag = dfx.mesh.MeshTags(mesh, cdim, cell_indices[sorted_cells], cell_markers[sorted_cells])

# Define function spaces
PHI_E = dfx.fem.FunctionSpace(mesh, ('CG',1))
PHI_S = PHI_E.clone()
LM_IAPP = PHI_E.clone()
phi_e, phi_s, lm_iapp = dfx.fem.Function(PHI_E, name='phi_e'), dfx.fem.Function(PHI_S, name='phi_s'), dfx.fem.Function(LM_IAPP,name='lm_iapp')
dphi_e, dphi_s, dlm_iapp = ufl.TrialFunction(PHI_E), ufl.TrialFunction(PHI_S), ufl.TrialFunction(LM_IAPP)
v_phi_e, v_phi_s, v_lm_iapp = ufl.TestFunction(PHI_E), ufl.TestFunction(PHI_S), ufl.TestFunction(LM_IAPP)
# Define measures
dx = ufl.Measure('dx', mesh, subdomain_data=cell_tag)
ds = ufl.Measure('ds', mesh, subdomain_data=facet_tag)
# Define constants 
I_app = dfx.fem.Constant(mesh, ScalarType(1))
V_app = dfx.fem.Constant(mesh, ScalarType(1))
switch = dfx.fem.Constant(mesh, ScalarType(0))


# Define WF
reaction = 1e-5*ufl.sinh(0.5*94685/(8.3*298)*(phi_s-phi_e))
residuals = [
    ufl.inner(ufl.grad(phi_e),ufl.grad(v_phi_e)) * dx - lm_iapp*v_phi_e * ds(2) - reaction*v_phi_e*dx(1),
    1e2*ufl.inner(ufl.grad(phi_s),ufl.grad(v_phi_s)) * dx(1) + reaction*v_phi_s*dx(1),
    (1-switch)*(I_app-lm_iapp)*v_lm_iapp*ds(2) + switch*(V_app-phi_e)*v_lm_iapp*ds(2)
]
jacobian = [[ufl.derivative(r_i, u_i, du_i) for u_i, du_i in zip([phi_e, phi_s, lm_iapp],[dphi_e, dphi_s, dlm_iapp])] for r_i in residuals]
bc = dfx.fem.dirichletbc(dfx.fem.Constant(mesh, ScalarType(0)), dfx.fem.locate_dofs_topological(PHI_S, fdim, left_facets), PHI_S)
restrictions = [
    multiphenicsx.fem.DofMapRestriction(PHI_E.dofmap, np.arange(0, PHI_E.dofmap.index_map.size_local+PHI_E.dofmap.index_map.num_ghosts)),
    multiphenicsx.fem.DofMapRestriction(PHI_S.dofmap, dfx.fem.locate_dofs_topological(PHI_S, cdim, left_cells)), 
    multiphenicsx.fem.DofMapRestriction(LM_IAPP.dofmap, dfx.fem.locate_dofs_topological(LM_IAPP, fdim, right_facets))
]

# Solve
problem = NonlinearBlockProblem(residuals, [phi_e, phi_s, lm_iapp], [bc], jacobian, restrictions)
solver = NewtonBlockSolver(mesh.comm, problem)
solver.solve()

