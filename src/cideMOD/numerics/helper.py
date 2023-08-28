#
# Copyright (c) 2023 CIDETEC Energy Storage.
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
"""Helper functions to analyze numerical problems"""
import dolfinx as dfx
import multiphenicsx.fem
import time
import numpy as np
from scipy.sparse import csr_matrix, linalg as sla
from matplotlib import pyplot as plt
from tabulate import tabulate
import os


def analyze_jacobian(J, fields):
    J_lab = [[i for i in fields] for i in fields]
    abs_J = np.zeros((len(fields), len(fields)))
    times = np.zeros((len(fields), len(fields)))
    start_time = time.time()
    for i, F in enumerate(fields):
        for j, fld in enumerate(fields):
            try:
                reference = time.time()
                norm = dfx.fem.assemble_matrix(J[i, j]).norm()
                times[i, j] = time.time() - reference
                abs_J[i, j] = norm
            except Exception as e:
                pass
            J_lab[i][j] = "F_{}/d_{}".format(F, fld)
            print(J_lab[i][j], abs_J[i, j], times[i, j])
    print('Total time:', time.time() - start_time)
    start_time = time.time()
    J = multiphenicsx.fem.petsc.assemble_matrix_block(J)
    print('Multiphenics time:', time.time() - start_time)
    mat = J.mat()
    spm = csr_matrix(mat.getValuesCSR()[::-1], shape=mat.size)
    plt.spy(spm)
    plt.savefig('J.png')


def plot_jacobian(problem, x, J, save=True):
    plt.clf()
    problem._prepare_variable_indices(x)
    size = x.size - 1
    frontiers = [d.max() for d in problem.dofs]
    locations = [(d.max() + d.min()) / 2 for d in problem.dofs]
    fields = [sol.name for sol in problem._solutions]
    csmat = csr_matrix(J.getValuesCSR()[::-1], shape=J.size)
    mat_dict = csmat.todok()
    xy = np.array(list(mat_dict.keys()))
    vals = np.array(list(mat_dict.values()))
    plt.scatter(xy[:, 1], size - xy[:, 0], s=5, c=np.log10(np.abs(vals)), cmap='inferno')
    for line in frontiers:
        plt.axhline(y=size - line, linestyle='--', linewidth=0.5)
        plt.axvline(x=line, linestyle='--', linewidth=0.5)
    for i, (text, xloc) in enumerate(zip(fields, locations)):
        factor = 1.1 if i % 2 == 0 else 1.05
        plt.text(xloc, max(frontiers) * factor, text, fontsize=8, ha='center')
    plt.colorbar()
    plt.xlim([0, size])
    plt.ylim([0, size])
    plt.xticks([], [])
    plt.yticks([], [])
    j = 0
    while os.path.exists(f'J_{j}.png'):
        j += 1
    if save:
        plt.savefig(f'J_{j}.png')
    else:
        plt.show()


def print_diagonal_statistics(problem):
    restriction = ((problem._restriction, problem._restriction)
                   if problem._restriction is not None else None)
    J = multiphenicsx.fem.petsc.assemble_matrix_block(
        problem._J, bcs=problem._bcs, restriction=restriction)
    J.assemble()
    mat = csr_matrix(J.getValuesCSR()[::-1], shape=J.size)
    diag = mat.diagonal()
    fields = [u.name for u in problem._solutions]
    vals = [[fields[i], abs(diag[dl]).max(), abs(diag[dl]).min()]
            for i, dl in enumerate(problem.dofs)]
    print(f'--------------- Diagonal values ---------------')
    print(tabulate(vals, headers=['Fields', 'Max', 'Min'], showindex=False))
    print('--------------------- End ----------------------')


def estimate_condition_number(A):
    mat = csr_matrix(A.getValuesCSR()[::-1], shape=A.size)
    return np.linalg.cond(mat.todense())
    ew1, ev = sla.eigs(mat, which='LM')
    ew2, ev = sla.eigs(mat, which="SM")

    ew1 = abs(ew1)
    ew2 = abs(ew2)

    condA = ew1.max() / ew2.min()
    return condA
