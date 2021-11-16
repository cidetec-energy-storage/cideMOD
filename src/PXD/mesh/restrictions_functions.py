from dolfin import MeshFunction, Timer, cells, entities, facets
from multiphenics import MeshRestriction

def _subdomain_restriction_old(subdomains_names, mesh, subdomains, field_data):
        if not isinstance(subdomains_names, (tuple, list)):
            subdomains_names = [subdomains_names]
        subdomain_ids = [field_data[name] for name in subdomains_names if name in field_data.keys()]
        D = mesh.topology().dim()
        # Initialize empty restriction
        restriction = MeshRestriction(mesh, None)
        for d in range(D + 1):
            mesh_function_d = MeshFunction("bool", mesh, d)
            mesh_function_d.set_all(False)
            restriction.append(mesh_function_d)
        t = Timer('Restriction - subdomain - old')
        # Mark restriction mesh functions based on subdomain id
        for c in cells(mesh):
            if subdomains[c] in subdomain_ids:
                restriction[D][c] = True
                for d in range(D):
                    for e in entities(c, d):
                        restriction[d][e] = True
        # Return
        t.stop()
        return restriction

def _subdomain_restriction(subdomains_names, mesh, subdomains, field_data):
    D = mesh.topology().dim()
    # Initialize empty restriction
    restriction = MeshRestriction(mesh, None)
    for d in range(D + 1):
        mesh_function_d = MeshFunction("bool", mesh, d)
        mesh_function_d.set_all(False)
        restriction.append(mesh_function_d)
    t = Timer('Restriction - subdomain')
    if not isinstance(subdomains_names, (tuple, list)):
        subdomains_names = [subdomains_names]
    if all(isinstance(name, str) for name in subdomains_names):
        # Mark restriction mesh functions based on subdomain id
        subdomain_ids = [field_data[name] for name in subdomains_names if name in field_data.keys()]
        cels = [c for c in cells(mesh)]
        for id in subdomain_ids:
            for index in subdomains.where_equal(id):
                if index < len(cels):
                    restriction[D][cels[index]] = True
                    for d in range(D):
                        for e in entities(cels[index], d):
                            restriction[d][e] = True
    elif all(isinstance(name, MeshRestriction) for name in subdomains_names):
        subdomain_res = subdomains_names
        for d in range(len(restriction)):
            mask = restriction[d].array()
            for res in subdomain_res:
                mask = mask | res[d].array()
            restriction[d].set_values(mask)
    else:
        raise Exception('subdomains_names should be of type str or MeshRestriction')
    t.stop()
    # Return
    return restriction

def _boundary_restriction_old(subdomains_names, mesh, boundaries, field_data):
    if not isinstance(subdomains_names, (tuple, list)):
        subdomains_names = [subdomains_names]
    subdomain_ids = [field_data[name] for name in subdomains_names]
    D = mesh.topology().dim()
    # Initialize empty restriction
    restriction = MeshRestriction(mesh, None)
    for d in range(D + 1):
        mesh_function_d = MeshFunction("bool", mesh, d)
        mesh_function_d.set_all(False)
        restriction.append(mesh_function_d)
    # Mark restriction mesh functions based on subdomain id
    for c in facets(mesh):
        if boundaries[c] in subdomain_ids:
            restriction[D-1][c] = True
            for d in range(D-1):
                for e in entities(c, d):
                    restriction[d][e] = True
    # Return
    return restriction

def _boundary_restriction(subdomains_names, mesh, boundaries, field_data):
    if not isinstance(subdomains_names, (tuple, list)):
        subdomains_names = [subdomains_names]
    subdomain_ids = [field_data[name] for name in subdomains_names]
    D = mesh.topology().dim()
    # Initialize empty restriction
    restriction = MeshRestriction(mesh, None)
    for d in range(D + 1):
        mesh_function_d = MeshFunction("bool", mesh, d)
        mesh_function_d.set_all(False)
        restriction.append(mesh_function_d)
    # Mark restriction mesh functions based on subdomain id
    t = Timer('Restriction - boundary')
    fac = [f for f in facets(mesh)]
    for id in subdomain_ids:
        for index in boundaries.where_equal(id): # Only iterate facets of interest
            restriction[D-1][fac[index]] = True
            for d in range(D-1):
                for e in entities(fac[index], d):
                    restriction[d][e] = True
    # Return
    t.stop()
    return restriction

def _generate_interface_mesh_function_old(mesh, subdomains, field_data):
    sets = []
    if 'separator' in field_data:
        if 'anode' in field_data:
            sets.append((set((field_data['anode'],field_data['separator'])),  1))
            if 'negativeCC' in field_data:
                sets.append((set((field_data['anode'],field_data['negativeCC'])), 3))
        if 'cathode' in field_data:
            sets.append((set((field_data['cathode'],field_data['separator'])), 2))
            if 'positiveCC' in field_data:
                sets.append((set((field_data['cathode'],field_data['positiveCC'])), 4))
    
    D = mesh.topology().dim()
    mf = MeshFunction('size_t', mesh, D-1, 0)
    for f in facets(mesh):
        subdomains_ids_f = set(subdomains[c] for c in cells(f))
        assert len(subdomains_ids_f) in (1, 2)
        if len(subdomains_ids_f) == 2:
            for x in sets:
                if subdomains_ids_f == x[0]:
                    mf[f] = x[1]
    # Return
    return mf

def _generate_interface_mesh_function(mesh, subdomains, field_data):
    """
    Generates interface MeshFunction

    Returns
    -------
    MeshFunction
        1 for anode-separator interface
        2 for cathode-separator interface
        3 for anode-CC interface
        4 for cathode-CC interface
    """
    sets = []
    if 'separator' in field_data:
        if 'anode' in field_data:
            sets.append((set((field_data['anode'],field_data['separator'])),  1))
            if 'negativeCC' in field_data:
                sets.append((set((field_data['anode'],field_data['negativeCC'])), 3))
        if 'cathode' in field_data:
            sets.append((set((field_data['cathode'],field_data['separator'])), 2))
            if 'positiveCC' in field_data:
                sets.append((set((field_data['cathode'],field_data['positiveCC'])), 4))
    # print(field_data, sets)
    D = mesh.topology().dim()
    mf = MeshFunction('size_t', mesh, D-1, 0)
    for f in facets(mesh):
        subdomains_ids_f = set(subdomains[c] for c in cells(f))
        if len(subdomains_ids_f) == 2:
            for x in sets:
                if subdomains_ids_f == x[0]:
                    mf[f] = x[1]
    # Return
    return mf

def _interface_restriction(subdomain_names, mesh, subdomains, field_data):
    assert isinstance(subdomain_names, (list,tuple))
    subdomains_ids=[]
    for name_pair in subdomain_names:
        assert isinstance(name_pair, (list,tuple))
        assert len(name_pair) == 2
        subdomains_ids.append(set([ field_data[name] for name in name_pair if name in field_data.keys()]))
    
    D = mesh.topology().dim()
    # Initialize empty restriction
    restriction = MeshRestriction(mesh, None)
    for d in range(D + 1):
        mesh_function_d = MeshFunction("bool", mesh, d)
        mesh_function_d.set_all(False)
        restriction.append(mesh_function_d)
    # Mark restriction mesh functions based on subdomain ids (except the mesh function corresponding to dimension D, as it is trivially false)
    t = Timer('Restriction - interface')
    for f in facets(mesh):
        if set(subdomains.array()[f.entities(D)]) in subdomains_ids:
            restriction[D - 1][f] = True
            for d in range(D - 1):
                for e in entities(f, d):
                    restriction[d][e] = True
    t.stop()
    return restriction