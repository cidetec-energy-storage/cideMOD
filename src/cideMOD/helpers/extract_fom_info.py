#
# Copyright (c) 2022 CIDETEC Energy Storage.
#
# This file is part of cideMOD.
#
# cideMOD is free software: you can redistribute it and/or modify
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
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import numpy as np
import dolfin as df
from sys import exit

#######################################################
########         VARIABLES SOLUTION         ###########
#######################################################
def get_current_results(problem, label):
    """
    Create a dictionary with the variables results at the current time step.

    :param problem: Instance of class Problem or NDProblem.
    :type problem: cideMOD.problem.NDProblem

    :param label: String indicating which variables, scaled or unscaled, you desire to extact.
    :type label: str

    :return: Dictionary with the variables results at the current time step.
    :rtype: dict
    """

    if not (label == 'unscaled' or label == 'scaled'):
        print('[helpers/extract_fom_info/get_current_results] ERROR: Unrecognized label option.')
        exit()

    # Get lists of dofs in the interfaces
    if not 'interfaces_dofs' in problem.fom2rom['mesh'].keys():
        # Empty dictionary
        get_interfaces_dofs(problem)

    # Get subdomain dofs
    dofs_anode      = problem.fom2rom['mesh']['subdomain_dofs']['anode']
    dofs_cathode    = problem.fom2rom['mesh']['subdomain_dofs']['cathode']
    dofs_separator  = problem.fom2rom['mesh']['subdomain_dofs']['separator']
    if problem.fom2rom['areCC']:
        dofs_positiveCC = problem.fom2rom['mesh']['subdomain_dofs']['positiveCC']
        dofs_negativeCC = problem.fom2rom['mesh']['subdomain_dofs']['negativeCC']

    # Get variable domains dofs
    if problem.fom2rom['areCC']:
        dofs_phis = np.unique(np.concatenate((np.concatenate((np.concatenate((dofs_negativeCC, dofs_anode)), dofs_cathode)), dofs_positiveCC)))
    else:
        dofs_phis = np.unique(np.concatenate((dofs_anode, dofs_cathode)))
    dofs_ce   = np.unique(np.concatenate((np.concatenate((dofs_anode, dofs_separator)), dofs_cathode)))
    dofs_cs   = np.unique(np.concatenate((dofs_anode, dofs_cathode)))
    dofs_phie = dofs_ce
    dofs_jLi  = dofs_cs

    # Get interfaces dofs
    if problem.fom2rom['areCC']:
        dofs_anodeCC   = problem.fom2rom['mesh']['interfaces_dofs']['anodeCC']
        dofs_cathodeCC = problem.fom2rom['mesh']['interfaces_dofs']['cathodeCC']

    # Get FOM fields
    fields = problem.f_1._fields
    # problem.f_1 has the adimensional solution for the problem variables
    # problem.dim_variables is the dimensional version of problem.f_1

    # Declare results dictionary
    resultsDict = dict()

    # Fill results dictionary
    field_names = ['ce', 'phie', 'phis', 'jLi', 'cs']
    if problem.model_options.solve_thermal:
        field_names.append('T')
    if problem.model_options.solve_SEI:
        field_names.extend(['cSEI','jSEI','deltaSEI'])
    for key in field_names:
        if   key == "ce":
            ce = problem.f_1.c_e.vector()[:]
            resultsDict[key] = np.zeros(ce.shape)
            if label == 'unscaled':
                resultsDict[key][dofs_ce] = problem.nd_model.c_e_0 + problem.nd_model.delta_c_e_ref * ce[dofs_ce]
            else:
                resultsDict[key][dofs_ce] = ce[dofs_ce]
        elif key == "phie":
            phie = problem.f_1.phi_e.vector()[:]
            resultsDict[key] = np.zeros(phie.shape)
            if label == 'unscaled':
                resultsDict[key][dofs_phie] = problem.nd_model.phi_e_ref + problem.nd_model.liquid_potential * phie[dofs_phie]
            else:
                resultsDict[key][dofs_phie] = phie[dofs_phie]
        elif key == "phis":
            if problem.fom2rom['areCC']:
                phis = problem.f_1.phi_s.vector()[:] + problem.f_1.phi_s_cc.vector()[:]
                phis[dofs_anodeCC]   /= 2.0
                phis[dofs_cathodeCC] /= 2.0
            else:
                phis = problem.f_1.phi_s.vector()[:]
            resultsDict[key] = np.zeros(phis.shape)
            if label == 'unscaled':
                resultsDict[key][dofs_phis] = problem.nd_model.phi_s_ref + problem.nd_model.solid_potential * phis[dofs_phis]
            else:
                resultsDict[key][dofs_phis] = phis[dofs_phis]
        elif key == "jLi":
            # Loop in number of materials
            for i in range(1):
                idx_a = fields.index("j_Li_a"+str(i))
                idx_c = fields.index("j_Li_c"+str(i))
                as_a = 3.0*problem.cell.negative_electrode.active_materials[0].volumeFraction/problem.cell.negative_electrode.active_materials[0].particleRadius
                as_c = 3.0*problem.cell.positive_electrode.active_materials[0].volumeFraction/problem.cell.positive_electrode.active_materials[0].particleRadius
                jLi_a = problem.f_1[idx_a].vector()[:]
                jLi_c = problem.f_1[idx_c].vector()[:]
                resultsDict[key] = np.zeros(jLi_a.shape)
                if label == 'unscaled':
                    resultsDict[key][dofs_jLi] = problem.nd_model.I_0/problem.nd_model.L_0 * (jLi_a[dofs_jLi]/as_a + jLi_c[dofs_jLi]/as_c) / problem.F
                else:
                    resultsDict[key][dofs_jLi] = (jLi_a[dofs_jLi] + jLi_c[dofs_jLi]) / problem.F
        elif key == "cs":
            # Loop in number of materials
            for i in range(1):
                # Loop through SGM order
                for j in range(problem.SGM.order):
                    idx_a = fields.index("c_s_"+str(j)+"_a"+str(i))
                    idx_c = fields.index("c_s_"+str(j)+"_c"+str(i))
                    cs = np.zeros(problem.f_1[idx_a].vector()[:].shape[0])
                    cs[dofs_anode]   = problem.f_1[idx_a].vector()[:][dofs_anode]
                    cs[dofs_cathode] = problem.f_1[idx_c].vector()[:][dofs_cathode]
                    if j == 0:
                        for k in range(1, problem.SGM.order):
                            cs[dofs_anode]   -= problem.f_1[fields.index("c_s_"+str(k)+"_a"+str(i))].vector()[:][dofs_anode]
                            cs[dofs_cathode] -= problem.f_1[fields.index("c_s_"+str(k)+"_c"+str(i))].vector()[:][dofs_cathode]
                        resultsDict[key] = np.zeros(cs.shape)
                        if label == 'unscaled':
                            resultsDict[key][dofs_anode]   = problem.nd_model.c_s_a_max[i]*cs[dofs_anode]
                            resultsDict[key][dofs_cathode] = problem.nd_model.c_s_c_max[i]*cs[dofs_cathode]
                        else:
                            resultsDict[key][dofs_anode]   = cs[dofs_anode]
                            resultsDict[key][dofs_cathode] = cs[dofs_cathode]
                    else:
                        if label == 'unscaled':
                            cs[dofs_anode]   = problem.nd_model.c_s_a_max[i]*cs[dofs_anode]
                            cs[dofs_cathode] = problem.nd_model.c_s_c_max[i]*cs[dofs_cathode]
                        resultsDict[key] = np.concatenate((resultsDict[key], cs))
        
        elif key == "T":
            T = problem.f_1.temp.vector()[:]
            resultsDict[key] = np.zeros(T.shape)
            if label == 'unscaled':
                resultsDict[key] = problem.nd_model.T_ref + problem.nd_model.thermal_gradient * T
            else:
                resultsDict[key] = T

        elif key == "cSEI":
            SEI_model = problem.SEI_model_a if problem.SEI_model_a else problem.SEI_model_c
            # Loop in number of materials
            for i in range(1):
                resultsDict[key] = None
                # Loop through SGM order
                for j in range(SEI_model.SLagM.order):
                    cSEI = None
                    # Loop over SEI models
                    for SEI_model in [problem.SEI_model_a, problem.SEI_model_c]:
                        if not SEI_model:
                            continue
                        c_EC_index = fields.index(f"c_EC_{j}_{SEI_model.domain}{i}")
                        if cSEI is None:
                            cSEI = np.zeros(problem.f_1[c_EC_index].vector()[:].shape[0])
                        
                        dofs = dofs_anode if SEI_model.tag == 'anode' else dofs_cathode
                        if label =='unscaled':
                            c_EC_ref = problem.nd_model.c_sei_a if SEI_model.tag == 'anode' else problem.nd_model.c_sei_c
                            cSEI[dofs] = c_EC_ref[i] * problem.f_1[c_EC_index].vector()[:][dofs]
                        else:
                            cSEI[dofs] = problem.f_1[c_EC_index].vector()[:][dofs]

                    if resultsDict[key] is None:
                        resultsDict[key] = cSEI
                    else:
                        resultsDict[key] = np.concatenate((resultsDict[key], cSEI))

        elif key == "jSEI":
            # Loop in number of materials
            for i in range(1):
                resultsDict[key] = None
                # Loop over SEI models
                for SEI_model in [problem.SEI_model_a, problem.SEI_model_c]:
                    if not SEI_model:
                        continue
                    j_sei_index = fields.index(f"j_sei_{SEI_model.domain}{i}")
                    dofs = dofs_anode if SEI_model.tag == 'anode' else dofs_cathode
                    jLiSEI = problem.f_1[j_sei_index].vector()[:]

                    if resultsDict[key] is None:
                        resultsDict[key] = np.zeros(jLiSEI.shape)

                    if label == 'unscaled':
                        a_s = SEI_model.electrode.active_material[i].a_s
                        resultsDict[key][dofs] = problem.nd_model.I_0/problem.nd_model.L_0 * jLiSEI[dofs]/a_s/problem.F
                    else:
                        resultsDict[key][dofs] = jLiSEI[dofs]/problem.F
        
        elif key == "deltaSEI":
            # Loop in number of materials
            for i in range(1):
                resultsDict[key] = None
                # Loop over SEI models
                for SEI_model in [problem.SEI_model_a, problem.SEI_model_c]:
                    if not SEI_model:
                        continue
                    delta_sei_index = fields.index(f"delta_sei_{SEI_model.domain}{i}")
                    dofs = dofs_anode if SEI_model.tag == 'anode' else dofs_cathode
                    deltaSEI = problem.f_1[delta_sei_index].vector()[:]

                    if resultsDict[key] is None:
                        resultsDict[key] = np.zeros(deltaSEI.shape)

                    if label == 'unscaled':
                        delta_sei_ref = problem.nd_model.delta_sei_a if SEI_model.tag == 'anode' else problem.nd_model.delta_sei_c
                        resultsDict[key][dofs] = delta_sei_ref[i] * deltaSEI[dofs]
                    else:
                        resultsDict[key][dofs] = deltaSEI[dofs]
        else:
            raise NameError('Unknown required variable')

    # TODO: Think how to save different solution for materials. For now there is supposed to be a single material

    return resultsDict

def store_results(problem, label):
    """
    Save the current time step results to the fom2rom['results'] dictionary of the Problem class.

    :param problem: Instance of class Problem or NDProblem.
    :type problem: cideMOD.problem.NDProblem

    :param label: String indicating which variables, scaled or unscaled, you desire to extact.
    :type label: str
    """

    # Obtain results dictionary with the current time step variables solution
    resultsDict = get_current_results(problem, label)

    for key in resultsDict:
        problem.fom2rom['results'][key][:,problem.current_timestep] = resultsDict[key].copy()

    problem.current_timestep += 1


def initialize_results(problem, N):
    """
    Initialize arrays where the snapshots will be saved.

    :param problem: Instance of class Problem or NDProblem.
    :type problem: cideMOD.problem.NDProblem
    
    :param N: Number of snapshots expected to save.
    :type N: int
    """
    field_names = ['phis', 'phie', 'ce', 'cs', 'jLi']
    if problem.model_options.solve_thermal:
        field_names.append('T')
    if problem.model_options.solve_SEI:
        field_names.extend(['cSEI','jSEI','deltaSEI'])
        SEI_model = problem.SEI_model_a if problem.SEI_model_a else problem.SEI_model_c
    Nx = problem.f_1.phi_s.vector()[:].shape[0]
    for field in field_names:
        if field == 'cs':
            problem.fom2rom['results'][field] = np.zeros([problem.SGM.order*Nx, N+1])
        elif field == 'cSEI':
            problem.fom2rom['results'][field] = np.zeros([SEI_model.SLagM.order*Nx, N+1])
        else:
            problem.fom2rom['results'][field] = np.zeros([Nx, N+1])


def extend_results(problem, N):
    """
    Extend arrays where the snapshots will be saved.

    :param problem: Instance of class Problem or NDProblem.
    :type problem: cideMOD.problem.NDProblem
    
    :param N: Number of additional snapshots expected to save.
    :type N: int
    """
    field_names = ['phis', 'phie', 'ce', 'cs', 'jLi']
    if problem.model_options.solve_thermal:
        field_names.append('T')
    if problem.model_options.solve_SEI:
        field_names.extend(['cSEI','jSEI','deltaSEI'])
        SEI_model = problem.SEI_model_a if problem.SEI_model_a else problem.SEI_model_c
    Nx = problem.f_1.phi_s.vector()[:].shape[0]
    for field in field_names:
        if field == 'cs':
            problem.fom2rom['results'][field] = np.hstack( (problem.fom2rom['results'][field], np.zeros([problem.SGM.order*Nx, N+1])) )
        elif field == 'cSEI':
            problem.fom2rom['results'][field] = np.hstack( (problem.fom2rom['results'][field], np.zeros([SEI_model.SLagM.order*Nx, N+1])) )      
        else:
            problem.fom2rom['results'][field] = np.hstack( (problem.fom2rom['results'][field], np.zeros([Nx, N+1])) )


#####################################################
########         MESH INFORMATION         ###########
#####################################################
def extract_bmesh(mesh, boundaries):
    """
    Obtain boundary mesh and the references of its finite elements.

    :param mesh: Mesh of the domain.
    :type mesh: dolfin.cpp.generation.BoxMesh

    :param boundaries: MeshFunction containing the reference labels for the boundary faces.
    :type boundaries: dolfin.cpp.mesh.MeshFunctionSizet

    :return: Tuple containing\n
     * **boundary_mesh** *(dolfin.cpp.mesh.BoundaryMesh)* - Mesh of the domain boundaries.
     * **bmesh_refs** *(numpy.ndarray)* - MeshFunction containing the reference labels for the boundary faces.
    :rtype: tuple
    """

    # Extract boundary mesh
    boundary_mesh = df.BoundaryMesh(mesh, 'exterior')

    # pfaces return the parent face which correspond with each finite element of the boundary mesh
        # pfaces[i] gives the parent mesh face index which correspond with the i-th face of the boundary mesh
    pfaces = np.copy(boundary_mesh.entity_map(boundary_mesh.geometric_dimension()-1).array()) # bmesh_cell_to_global_edge_map

    # Create an array where set the boundary mesh elements references
    bmesh_refs = np.empty(pfaces.shape)

    # Write boundary mesh elements references from parent mesh boundary references
    for ct, k in enumerate(pfaces):
        bmesh_refs[ct] = boundaries.array()[k]

    return boundary_mesh, bmesh_refs

def get_mesh_info(problem, label):
    """
    Get neccesary mesh information from FOM model to construct ROM model.

    :param problem: Instance of class Problem or NDProblem.
    :type problem: cideMOD.problem.NDProblem
    
    :param label: String indicating which variables, scaled or unscaled, you desire to extact.
    :type label: str
    """

    # Create mesh dict results in fom2rom dict
    problem.fom2rom['mesh'] = dict()
    mesh = problem.fom2rom['mesh']

    ############################
    ####    MESH MAPPING    ####
    ############################

    # Conectivity matrix
    mat_connect = problem.mesher.mesh.cells()
    # mat_connect is the connectivity matrix with shape number of elements times number of dofs per finite element
        # mat_connect[i,:] gives the mesh vertex of the i-th finite element

    # Coordinates
    # dofs_coord = problem.mesher.mesh.coordinates()
    #     # dofs_coord[i,:] gives the coordinates of the i-th mesh dof
    if   label == 'unscaled':
        dofs_coord = problem.W[0].tabulate_dof_coordinates()*problem.nd_model.L_0
    elif label == 'scaled':
        dofs_coord = problem.W[0].tabulate_dof_coordinates()
    else:
        print('[helpers/extract_fom_info/get_mesh_info] ERROR: Unrecognized label option.')
        exit()
    mesh['dofs_coord'] = dofs_coord
        # dofs_coord[i,:] gives the coordinates of the i-th mesh dof
    ver2dof = df.vertex_to_dof_map(problem.W[0])
    # ver2dof is a vector that indicates the P1 dof index which corresponds to each mesh vertex.
        # ver2dof[i] gives the P1 dof global numeration which corresponds with the i-th mesh vertex
    dof2ver = df.dof_to_vertex_map(problem.W[0])
    # dof2ver is a vector that indicates the mesh vertex which corresponds to each P1 dof index.
        # dof2ver[i] gives the mesh vertex which corresponds with the i-th P1 dof global numeration
    mesh['ver2dof'] = ver2dof
    mesh['dof2ver'] = dof2ver
    dofs_index = ver2dof[mat_connect]
    # P1 dofs index for all finite elements
    mesh['dofs_index'] = dofs_index
        # dofs_index[i,:] gives the dofs of the i-th finite element


    #####################################################
    ####    FINITE ELEMENT SUBDOMAIN - REFERENCES    ####
    #####################################################

    # Vertex/Cell subdomain index
    domain_refs = problem.mesher.field_data
    problem.fom2rom['areCC'] = False
    if any(k in problem.cell.structure for k in ['pcc','ncc']):
        problem.fom2rom['areCC'] = True
        # Dictionary with the references of each subdomain
    elem_subdomain_ref = problem.mesher.subdomains.array()
    mesh['elem_subdomain_ref'] = elem_subdomain_ref
        # elem_subdomain_ref[i,:] gives the subdomain reference of the i-th mesh finite element
    mesh['subdomain_elems'] = {}
    mesh['subdomain_elems']['anode']      = np.where(elem_subdomain_ref==domain_refs['anode'])[0]
    mesh['subdomain_elems']['cathode']    = np.where(elem_subdomain_ref==domain_refs['cathode'])[0]
    mesh['subdomain_elems']['separator']  = np.where(elem_subdomain_ref==domain_refs['separator'])[0]
    if problem.fom2rom['areCC']:
        mesh['subdomain_elems']['negativeCC'] = np.where(elem_subdomain_ref==domain_refs['negativeCC'])[0]
        mesh['subdomain_elems']['positiveCC'] = np.where(elem_subdomain_ref==domain_refs['positiveCC'])[0]
    # subdomain_elems is a dictionary which contains the finite elements in each domain


    ################################
    ####     SUBDOMAIN DOFs     ####
    ################################
    mesh['subdomain_dofs'] = {}
    mesh['subdomain_dofs']['anode']      = problem.P1_map.domain_dof_map['anode']
    mesh['subdomain_dofs']['cathode']    = problem.P1_map.domain_dof_map['cathode']
    mesh['subdomain_dofs']['separator']  = problem.P1_map.domain_dof_map['separator']
    if problem.fom2rom['areCC']:
        mesh['subdomain_dofs']['positiveCC'] = problem.P1_map.domain_dof_map['positiveCC']
        mesh['subdomain_dofs']['negativeCC'] = problem.P1_map.domain_dof_map['negativeCC']
    # subdomain_dofs is a dictionary which contains the dof index which belongs to each domain


    #####################################
    ####    BOUNDARY MESH MAPPING    ####
    #####################################
        # bmesh: boundary mesh
        # bmesh_refs: reference of each finite element of the boundary mesh
    bmesh, bmesh_refs = extract_bmesh(problem.mesher.mesh, problem.mesher.boundaries)
    # Create a list of the finite elements in the CC
    fin_elem_pp = []
    fin_elem_np = []
    for i in range(bmesh_refs.shape[0]):
        if bmesh_refs[i] == domain_refs['positivePlug']:
            fin_elem_pp.append(i)
        elif bmesh_refs[i] == domain_refs['negativePlug']:
            fin_elem_np.append(i)
    # Convert the list into numpy arrays
    fin_elem_pp = np.array(fin_elem_pp)
    fin_elem_np = np.array(fin_elem_np)

    bmesh_mat_connect = bmesh.cells()
        # bmesh_mat_connect[i,:] gives the boundary mesh vertex of the i-th finite element of the boundary mesh

    # Connectivity matrix for finite element with references nP and PP. Return the vertex index of each finite element
    refPP_bmesh_mat_connect = bmesh_mat_connect[fin_elem_pp]
    refNP_bmesh_mat_connect = bmesh_mat_connect[fin_elem_np]


    ###############################################
    ####    BOUNDARY MESH AND MESH RELATION    ####
    ###############################################

    # bmesh_vertex_to_mesh_vertex is an 1D array that indicates which vertex of the parent mesh corresponds to each boundary mesh vertex
        # bmesh_vertex_to_mesh_vertex[i] gives the vertex index in the parent mesh which corresponds with the i-th boundary mesh vertex
    bmesh_vertex_to_mesh_vertex = bmesh.entity_map(0).array()

        # Vertex of the parent mesh of each boundary mesh finite element
    vertex_index_PP = bmesh_vertex_to_mesh_vertex[refPP_bmesh_mat_connect]
    vertex_index_NP = bmesh_vertex_to_mesh_vertex[refNP_bmesh_mat_connect]

        # dofs of the parent mesh of each boundary mesh finite element
    mesh['boundary'] = dict()
    mesh['boundary']['dofs_index'] = ver2dof[bmesh_vertex_to_mesh_vertex[bmesh_mat_connect]]

    dofs_index_PP = ver2dof[vertex_index_PP]
    mesh['boundary']['dofs_index_PP'] = dofs_index_PP
    dofs_index_NP = ver2dof[vertex_index_NP]
    mesh['boundary']['dofs_index_NP'] = dofs_index_NP

    return

def get_interfaces_dofs(problem):
    """
    Obtain 1D arrays containing the dofs that are in thge subdomain interfaces.

    :param problem: Instance of class Problem or NDProblem.
    :type problem: cideMOD.problem.NDProblem
    """

    ver2dof = df.vertex_to_dof_map(problem.W[0])
    # ver2dof is a vector that indicates the P1 dof index which corresponds to each mesh vertex.
        # ver2dof[i] gives the P1 dof global numeration which corresponds with the i-th mesh vertex

    domain_refs = problem.mesher.field_data # Dictionary with the references of each subdomain and interface
    facets_refs = problem.mesher.interfaces.array() # 2D mesh element references
    mfacets = df.facets(problem.mesher.mesh) # FacetIterator

    vertex_cathodeCC         = [] # Initialized list of vertex in cathode-CC        interface
    vertex_anodeCC           = [] # Initialized list of vertex in anode-CC          interface
    vertex_anode_separator   = [] # Initialized list of vertex in anode-separator   interface
    vertex_cathode_separator = [] # Initialized list of vertex in cathode-separator interface
    count = 0
    for facet in mfacets:
        if facets_refs[count] == domain_refs['interfaces']['cathode-CC']:
            aux = facet.entities(0) # Return vertex of the current mfacets
            for i in range(aux.shape[0]):
                vertex_cathodeCC.append(aux[i])
        elif facets_refs[count] == domain_refs['interfaces']['anode-CC']:
            aux = facet.entities(0) # Return vertex of the current mfacets
            for i in range(aux.shape[0]):
                vertex_anodeCC.append(aux[i])
        elif facets_refs[count] == domain_refs['interfaces']['anode-separator']:
            aux = facet.entities(0) # Return vertex of the current mfacets
            for i in range(aux.shape[0]):
                vertex_anode_separator.append(aux[i])
        elif facets_refs[count] == domain_refs['interfaces']['cathode-separator']:
            aux = facet.entities(0) # Return vertex of the current mfacets
            for i in range(aux.shape[0]):
                vertex_cathode_separator.append(aux[i])
        count += 1

    ###############################
    ####    DOFs INTERFACES    ####
    ###############################
    # Initialize dictionary where save the results
    problem.fom2rom['mesh']['interfaces_dofs'] = dict()

    if problem.fom2rom['areCC']:
        # anode-CC
        problem.fom2rom['mesh']['interfaces_dofs']['anodeCC'] = ver2dof[np.unique(np.array(vertex_anodeCC,dtype=int))]
            # Array containing the list of dofs in the anode-CC interface

        # cathode-CC
        problem.fom2rom['mesh']['interfaces_dofs']['cathodeCC'] = ver2dof[np.unique(np.array(vertex_cathodeCC,dtype=int))]
            # Array containing the list of dofs in the cathode-CC interface

    # anode-separator
    problem.fom2rom['mesh']['interfaces_dofs']['anode_separator'] = ver2dof[np.unique(np.array(vertex_anode_separator))]
        # Array containing the list of dofs in the anode-separator interface

    # cathode-separator
    problem.fom2rom['mesh']['interfaces_dofs']['cathode_separator'] = ver2dof[np.unique(np.array(vertex_cathode_separator))]
        # Array containing the list of dofs in the cathode-separator interface

    return

def get_spectral_info(problem):
    """
    Obtain the necessary pre-processed information about the spectral methods used in the model.

    :param problem: Instance of class Problem or NDProblem.
    :type problem: cideMOD.problem.NDProblem
    """
    problem.fom2rom['SGM'] = dict()

    # Get SGM matrices of the Li mass transport equation in the active particles
    problem.fom2rom['SGM']['cs'] = dict()
    problem.fom2rom['SGM']['cs']['M'] = problem.SGM.M
    problem.fom2rom['SGM']['cs']['K'] = problem.SGM.K
    problem.fom2rom['SGM']['cs']['P'] = problem.SGM.P

    # Get SGM matrices of the electrolyte solvent transport in the SEI
    if problem.model_options.solve_SEI:
        problem.fom2rom['SGM']['cSEI'] = dict()
        SEI_model = problem.SEI_model_a if problem.SEI_model_a else problem.SEI_model_c
        if SEI_model:
            problem.fom2rom['SGM']['cSEI']['f'] = SEI_model.SLagM.f
            problem.fom2rom['SGM']['cSEI']['D'] = SEI_model.SLagM.D
            problem.fom2rom['SGM']['cSEI']['K1'] = SEI_model.SLagM.K1
            problem.fom2rom['SGM']['cSEI']['K2'] = SEI_model.SLagM.K2
            problem.fom2rom['SGM']['cSEI']['P'] = SEI_model.SLagM.P
