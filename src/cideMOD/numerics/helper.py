"""Helper functions to analyze numerical problems"""
import dolfinx as dfx
import multiphenicsx.fem
import time
import numpy as np
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt
from tabulate import tabulate
import os

def analyze_jacobian(J, fields):
    J_lab = [[i for i in fields] for i in fields]
    abs_J = np.zeros((len(fields),len(fields)))
    times = np.zeros((len(fields),len(fields)))
    start_time = time.time()
    for i, F in enumerate(fields):
        for j, fld in enumerate(fields):
            try:
                reference = time.time()
                norm = dfx.fem.assemble_matrix(J[i,j]).norm()
                times[i,j] = time.time()-reference
                abs_J[i,j] = norm
            except Exception as e:
                pass
            J_lab [i][j] = "F_{}/d_{}".format(F,fld)
            print(J_lab[i][j], abs_J[i,j], times[i,j])
    print('Total time:',time.time()-start_time) 
    start_time = time.time()
    J = multiphenicsx.fem.petsc.assemble_matrix_block(J) 
    print('Multiphenics time:',time.time()-start_time)  
    mat = J.mat()
    spm = csr_matrix(mat.getValuesCSR()[::-1], shape=mat.size)
    plt.spy(spm)
    plt.savefig('J.png')

def plot_jacobian(problem, x, J):
    plt.clf()
    problem._prepare_variable_indices(x)
    size = x.size-1
    frontiers = [d.max() for d in problem.dofs]
    locations = [(d.max()+d.min())/2 for d in problem.dofs]
    fields = [sol.name for sol in problem._solutions]
    csmat = csr_matrix(J.getValuesCSR()[::-1],shape=J.size)
    mat_dict = csmat.todok()
    xy = np.array(list(mat_dict.keys()))
    vals = np.array(list(mat_dict.values()))
    plt.scatter(xy[:,1],size-xy[:,0],s=5, c=np.log10(np.abs(vals)),cmap='inferno')
    for line in frontiers:
        plt.axhline(y=size-line, linestyle='--',linewidth=0.5)
        plt.axvline(x=line, linestyle='--',linewidth=0.5)
    for i, (text, xloc) in enumerate(zip(fields, locations)):
        factor = 1.1 if i%2==0 else 1.05
        plt.text(xloc, max(frontiers)*factor, text, fontsize=8, ha='center')
    plt.colorbar()
    plt.xlim([0,size])
    plt.ylim([0,size])
    plt.xticks([],[])
    plt.yticks([],[])
    j=0
    while os.path.exists(f'J_{j}.png'):
        j+=1
    plt.savefig(f'J_{j}.png')

def print_diagonal_statistics(problem, inverse=False):
    restriction = None if problem._restriction is None else (problem._restriction, problem._restriction)
    J = multiphenicsx.fem.petsc.assemble_matrix_block(problem._J, bcs=problem._bcs,restriction=restriction)
    J.assemble()
    mat = csr_matrix(J.getValuesCSR()[::-1], shape=J.size)
    diag = mat.diagonal()
    fields = [u.name for u in problem._solutions]
    vals = [[fields[i], diag[dl].max(), diag[dl].min()] for i, dl in enumerate(problem.dofs)]
    print(f'--------------- Diagonal values ---------------')
    print(tabulate(vals, headers=['Fields', 'Max', 'Min'],showindex=False))
    print('--------------------- End ----------------------')
    if inverse:
        F = multiphenicsx.fem.petsc.assemble_vector_block(problem._F, problem._J, problem._bcs, scale=-1.0,restriction=problem._restriction, restriction_x0=problem._restriction)
        F.assemble()
        inverse_mat = np.linalg.pinv(mat.toarray())
        delta_x = inverse_mat@F.array
        vals = [[fields[i], delta_x[dl].max(), delta_x[dl].min()] for i, dl in enumerate(problem.dofs)]
        print(f'--------------- Step vector ---------------')
        print(tabulate(vals, headers=['Fields', 'Max', 'Min'],showindex=False))
        print('------------------- End --------------------')
