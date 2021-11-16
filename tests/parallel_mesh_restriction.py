from dolfin import *
from multiphenics import *
from numpy import array, zeros

parameters['ghost_mode']='shared_facet'

def generate_subdomain_restriction(mesh, subdomains, subdomain_id):
    D = mesh.topology().dim()
    # Initialize empty restriction
    restriction = MeshRestriction(mesh, None)
    for d in range(D + 1):
        mesh_function_d = MeshFunction("bool", mesh, d)
        mesh_function_d.set_all(False)
        restriction.append(mesh_function_d)
    # Mark restriction mesh functions based on subdomain id
    for c in cells(mesh):
        if subdomains[c] == subdomain_id:
            restriction[D][c] = True
            for d in range(D):
                for e in entities(c, d):
                    restriction[d][e] = True
    # Return
    return restriction

# Helper function to generate interface restriction based on a pair of gmsh subdomain ids
def generate_interface_restriction(mesh, subdomains, subdomain_ids):
    assert isinstance(subdomain_ids, set)
    assert len(subdomain_ids) == 2
    D = mesh.topology().dim()
    # Initialize empty restriction
    restriction = MeshRestriction(mesh, None)
    for d in range(D + 1):
        mesh_function_d = MeshFunction("bool", mesh, d)
        mesh_function_d.set_all(False)
        restriction.append(mesh_function_d)
    # Mark restriction mesh functions based on subdomain ids (except the mesh function corresponding to dimension D, as it is trivially false)
    for f in facets(mesh):
        subdomains_ids_f = set(subdomains[c] for c in cells(f))
        assert len(subdomains_ids_f) in (1, 2)
        if subdomains_ids_f == subdomain_ids:
            restriction[D - 1][f] = True
            for d in range(D - 1):
                for e in entities(f, d):
                    restriction[d][e] = True
    # Return
    return restriction

def get_local_dofs_on_restriction(V, restriction):
    """
    Computes dofs of W[component] which are on the provided restriction, which can be smaller or equal to the restriction
    provided at construction time of W (or it can be any restriction if W[component] is unrestricted). Returns two lists:
    * the first list stores local dof numbering with respect to W[component], e.g. to be used to fetch data
    from FEniCS solution vectors.
    * the second list stores local dof numbering with respect to W, e.g. to be used to fetch data from
    multiphenics solution block_vector.
    """
    # Prepare an auxiliary block function space, restricted on the boundary
    W_restricted = BlockFunctionSpace([V], restrict=[restriction])
    component_restricted = 0 # there is only one block in the W_restricted space
    # Get list of all local dofs on the restriction, numbered according to W_restricted. This will be a contiguous list
    # [1, 2, ..., # local dofs on the restriction]
    restricted_dofs = W_restricted.block_dofmap().block_owned_dofs__local_numbering(component_restricted)
    # Get the mapping of local dofs numbering from W_restricted[0] to V
    restricted_to_original = W_restricted.block_dofmap().block_to_original(component_restricted)
    # Get list of all local dofs on the restriction, but numbered according to V. Note that this list will not be
    # contiguous anymore, because there are DOFs on V other than the ones in the restriction (i.e., the ones in the
    # interior)
    original_dofs = [restricted_to_original[restricted] for restricted in restricted_dofs]
    return original_dofs

class SubdomainMapper:
    def __init__(self, field_data, function_space):
        t = Timer('Build SubdomainMapper')
        self.domain_vertex_map = {}
        self.domain_dof_map = {}
        self.ow_range = function_space.dofmap().ownership_range()
        self.base_array = zeros(len(function_space.dofmap().dofs()))
        self.base_function = Function(function_space)
        # print(self.ow_range)
        for field_name, res in field_data.items():
            self.domain_dof_map[field_name] = get_local_dofs_on_restriction(function_space, res)
        t.stop()

    def generate_vector(self, source_dict:dict):
        out_array = self.base_array.copy()
        for domain_name, source in source_dict.items():
            if domain_name in self.domain_dof_map.keys():
                if isinstance(source, (float, int)):
                    out_array[self.domain_dof_map[domain_name]] = source
                else:
                    raise Exception('Invalid source type for domain mapper')
        return out_array

    def generate_function(self, source_dict:dict):
        ou_funct = self.base_function.copy(deepcopy=True) 
        vec = ou_funct.vector()
        vec.set_local(self.generate_vector(source_dict))
        vec.apply("insert")
        return ou_funct

m = UnitCubeMesh(20,20,20)

left = CompiledSubDomain("x[0]<=0.3+tol",tol=DOLFIN_EPS)
right = CompiledSubDomain("x[0]>=0.3-tol",tol=DOLFIN_EPS)
intf = CompiledSubDomain("near(x[0],0.3)")


subs = MeshFunction('size_t', m, m.topology().dim(),0)
left.mark(subs, 1)
right.mark(subs, 2)
bounds = MeshFunction('size_t', m, m.topology().dim()-1,0)
intf.mark(bounds, 1)

lres = MeshRestriction(m,left)
rres = MeshRestriction(m,right)
ires = MeshRestriction(m, intf)

# Generate restrictions doesn't work in parallel for np>2
# lres = generate_subdomain_restriction(m, subs, 1)
# rres = generate_subdomain_restriction(m, subs, 2)
# ires = generate_interface_restriction(m, subs, set((1,2)))

field_data = {'left':lres, 'right':rres, 'interface':ires}

P1 = FunctionSpace(m, 'CG', 1)

W = BlockFunctionSpace([P1,P1,P1], restrict=[lres,rres,ires])

f = BlockFunction(W)
test = BlockTestFunction(W)

maper = SubdomainMapper(field_data, P1)
print('left:',len(maper.domain_dof_map.get('left',[])),'right:',len(maper.domain_dof_map.get('right',[])),'int:',len(maper.domain_dof_map.get('interface',[])))

u,v,w = block_split(f)
u.assign(maper.generate_function({'left':100}))
v.assign(maper.generate_function({'right':200}))
w.assign(interpolate(Expression("0.5*x[1]",degree=1),P1))

uh, vh, wh = block_split(test)

# XDMFFile('test.xdmf').write(f.sub(0))
# XDMFFile('test2.xdmf').write(f.sub(1))

dxl = Measure('dx',domain=m,subdomain_data=subs, subdomain_id=1)
dxr = Measure('dx',domain=m,subdomain_data=subs, subdomain_id=2)
dxi = Measure('dS',domain=m,subdomain_data=bounds, subdomain_id=1)

print('stiffnes left' ,assemble(inner(grad(u), grad(uh))*dxl).norm('l1'))
print('stiffnes right' ,assemble(inner(grad(v), grad(vh))*dxr).norm('l1'))
print('stiffnes int+' ,assemble(inner(grad(w), grad(wh))('+')*dxi).norm('l1')) # This should be 1
print('stiffnes int-' ,assemble(inner(grad(w), grad(wh))('-')*dxi).norm('l1')) # This should be 1
