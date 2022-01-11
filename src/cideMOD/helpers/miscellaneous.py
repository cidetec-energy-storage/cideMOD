#
# Copyright (c) 2021 CIDETEC Energy Storage.
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
"""miscellaneous.py involves all the auxiliary functions which
dosen't belong to any class.

Functions
---------
get_spline(data, spline_type = "not-a-knot")

"""
import dolfin as df
import multiphenics as mph

import glob
import json
import os
import pathlib
import shutil
import time

import matplotlib.pyplot as plt
import numpy
from numpy.polynomial.polynomial import *
from scipy.interpolate import CubicSpline
from scipy.sparse import csr_matrix


def get_spline(data, spline_type = "not-a-knot", return_fenics = True):
    """This function adapts the CubicSpline of scipy package to
    the UFL classes type.

    It gets the spline coefficients and uses them for computing
    the spline in a given point, y.

    Parameters
    ----------
    data : array [x_vector, y_vector]
        Data in array form, the first column is the x values,
        and the second column is the function value for x.
    spline_type : str, optional
        Spline type for the scipy.interpolate.CubicSpline. See
        scipy documentation for more types, by default "not-a-knot".

    Returns
    -------
    UFL Expression
        Spline expression.

    Raises
    ------
    NameError
        'Unknown type of interpolation'
    """
    data = numpy.array(data)
    if data[0,0]>data[-1,0]:
        data=numpy.flip(data,0)
    x_array = data[:,0]
    v_array = data[:,1]

    try:
        spline = CubicSpline(x_array, v_array, bc_type = spline_type)
        c = spline.c
        k = len(c) - 1

    except Exception as e:
        print(e)
        raise NameError('Unknown type of interpolation')
    if return_fenics:
        def f(y):
            S_list = []

            for j in range(len(x_array)-1):
                S_list.append(sum(c[m, j] * (y - x_array[j])**(k-m) for m in range(k+1)))

            fy = 0

            for j in range(len(S_list)):
                fy += S_list[j] * df.conditional(df.ge(y,x_array[j]), df.conditional(df.lt(y,x_array[j+1]), 1, 0), 0)

            fy += S_list[+0]*df.conditional(df.lt(y,x_array[+0]), 1, 0)
            fy += S_list[-1]*df.conditional(df.ge(y,x_array[-1]), 1, 0)

            return fy
        return f
    else:
        return spline

def hysteresys_property(property:dict):
    def value_selector(x, current = None):
        if current:
            return df.conditional(df.gt(current, 0), property['charge'](x), df.conditional(df.lt(current, 0), property['discharge'](x), property['charge'](x)*0.5+property['discharge'](x)*0.5) )
        else:
            return property['discharge'](x)
    return value_selector

def init_results_folder(case_name,overwrite=False, copy_files:list=[]):
    """Function which initialize the results folder.

    Parameters
    ----------
    case_name : str
        String containing the case name.
    overwrite : bool, optional
        Switch for overwritting an existing case_name,
        by default False.

    Returns
    -------
    str
        Complete saving path.
    """
    save_path = 'results_{}'.format(case_name)
    comm = df.MPI.comm_world
    if df.MPI.rank(comm)==0:
        if not overwrite:
            i=1
            while os.path.exists(save_path):
                save_path='results_{}_v{}'.format(case_name,i)
                i=i+1
    save_path = comm.bcast(save_path,root=0)
    if df.MPI.rank(comm) == 0:
        try:
            os.stat(save_path); shutil.rmtree(save_path); os.mkdir(save_path)
        except:
            os.mkdir(save_path)
        print('Saving results to',os.path.realpath(save_path))
        df.set_log_level(df.LogLevel.WARNING)
        for i, cpfile in enumerate(copy_files):
            if isinstance(cpfile, str):
                path = pathlib.Path(cpfile)
                if path.exists():
                    shutil.copy(path,save_path)
            elif isinstance(cpfile, dict):
                with open(os.path.join(save_path,f'conf_{i}.json'),'w') as fout:
                    json.dump(cpfile,fout,indent=4,sort_keys=True)
    else:
        df.set_log_level(df.LogLevel.ERROR)
    return save_path

def constant_expression(expression, **kwargs):
    """Evaluates expression with given arguments

    Parameters
    ----------
    expresion : str
        String form of the expression in python syntax.
    **kwargs: 
        variables to replace inside the expresion.
    
    Returns
    -------
    value: Float, Constant
        Evaluation of the expression or constant given.
    """
    if isinstance(expression,str):
        value = eval(expression,{**kwargs,**df.__dict__})
    else:
        value = expression
    return value

def inside_element_expression(list):
    expression = ""
    for i_index, e_index in enumerate(list):
        expression += "(x[0] >= " + str(e_index) + " - tol && x[0] <= " + str(e_index + 1) + " + tol)"
        if i_index < (len(list) - 1):
            expression += " || "
    if expression == "":
        expression = "false"
    return expression

def plot_list_variable(x, y, name, direc, show=False, hide_ax_tick_labels=False,
                    label_axes=True, title='', hide_axis=False, xlabel = 'x',
                    ylabel = 'y', ymin=None, ymax=None, xmin = None, xmax = None,
                    i_app = None, data_path = None, save = True, ref=""):
    """This function plots y values in x axis.

    Parameters
    ----------
    x : list
        List values for x axis.
    y : list
        List values for y axis.
    name : str
        Saving image file name.
    direc : str
        Saving path.
    show : bool, optional
        Switch for showing the figure, by default True
    hide_ax_tick_labels : bool, optional
        Switch for hiding the axis tick labels, by default False
    label_axes : bool, optional
        Switch for writting the axis labels, by default True
    title : str, optional
        Plot tittle, by default ''
    hide_axis : bool, optional
        Switch for hiding the axis, by default False
    xlabel : str, optional
        X label name, by default 'x'
    ylabel : str, optional
        Y label name, by default 'y'
    """

    d    = os.path.dirname(direc)
    if not os.path.exists(d):
        os.makedirs(d)

    fig = plt.figure(figsize=(5,5))
    ax  = fig.add_subplot(111)

    p = ax.plot(x, y, '-', lw=1, label="Simulation")
    if i_app:
        if "Time" in xlabel:
            try:
                path_list = glob.glob('./' + data_path + '/t_V_'+str(i_app)+'C*'+str(ref)+'.txt', recursive=False)
                data = read_input_data(path_list[0], init_line=1)
                xp = data[:,0]
                yp = data[:,1]
                p = ax.plot(xp, yp, '.', lw=1, label="Validation data")
                plt.legend()
            except:
                pass
        if "Capacity" in xlabel:
            try:
                path_list = glob.glob('./' + data_path + '/C_V_'+str(i_app)+'C*'+str(ref)+'.txt', recursive=False)
                data = read_input_data(path_list[0], init_line=1)
                xp = data[:,0]
                yp = data[:,1]
                p = ax.plot(xp, yp, '.', lw=1, label="Validation data")
                plt.legend()
            except:
                pass

    if ymin == None:
        ymin = min(y)
    if ymax == None:
        ymax = max(y)
    if xmin == None:
        xmin = min(x)
    if xmax == None:
        xmax = max(x)

    Dy = (ymax - ymin) * 0.05
    Dx = (xmax - xmin) * 0.05

    ax.set_xlim([xmin - Dx, xmax + Dx])

    ax.set_ylim([ymin - Dy, ymax + Dy])

    if label_axes:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    if hide_ax_tick_labels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    if hide_axis:
        plt.axis('off')

    tit = plt.title(title)
    plt.tight_layout()
    if save:
        plt.savefig(direc + name + '.png', dpi=500)

    if show:
        plt.show()
        plt.close(fig)

def read_input_data(filename, init_line = 0):
    fichero = open(filename,'r')
    f = []
    line = '0'
    i = 0

    while i < init_line:
        i += 1
        fichero.readline()

    while len(line)>0:
        line = numpy.array(fichero.readline().split()).astype(float)
        if len(line)>0 :
            f.append(line)
    fichero.close()
    return numpy.array(f)

def analyze_jacobian(J, fields):
    J_lab = [[i for i in fields] for i in fields]
    abs_J = numpy.zeros((len(fields),len(fields)))
    times = numpy.zeros((len(fields),len(fields)))
    start_time = time.time()
    for i, F in enumerate(fields):
        for j, fld in enumerate(fields):
            try:
                reference = time.time()
                norm = df.assemble(J[i,j]).norm('l1')
                times[i,j] = time.time()-reference
                abs_J[i,j] = norm
            except Exception as e:
                pass
            J_lab [i][j] = "F_{}/d_{}".format(F,fld)
            print(J_lab[i][j], abs_J[i,j], times[i,j])
    print('Total time:',time.time()-start_time) 
    start_time = time.time()
    J = mph.block_assemble(J) 
    print('Multiphenics time:',time.time()-start_time)  
    mat = J.mat()
    spm = csr_matrix(mat.getValuesCSR()[::-1], shape=mat.size)
    plt.spy(spm)
    plt.savefig('J.png')

def format_time(timespan, small=True):
    # we have more than a minute, format that in a human readable form
    # Idea from http://snipplr.com/view/5713/
    if small:
        parts = [("d", 60*60*24),("h", 60*60),("min", 60), ("s", 1)]
    else:
        parts = [(" days", 60*60*24),(" hours", 60*60),(" minutes", 60), (" seconds", 1)]
    time = []
    leftover = timespan
    for suffix, length in parts:
        value = int(leftover / length)
        if value > 0:
            leftover = leftover % length
            time.append(u'%s%s' % (str(value), suffix))
        if leftover < 1:
            break
    return " ".join(time)
            
def create_problem_json(problem_dic, V):
    problem_dic_json = dict()
    for key in problem_dic.keys():
         if type(problem_dic[key]) in [str, float, int, bool]:
             problem_dic_json[key] = problem_dic[key]
         elif type(problem_dic[key]) is list:
             if not problem_dic[key]:
                 problem_dic_json[key] = problem_dic[key]
             elif problem_dic[key][0] is None:
                 problem_dic_json[key] = problem_dic[key]
             elif problem_dic[key][0] in [str, float, int, bool]:
                 problem_dic_json[key] = problem_dic[key]
             elif problem_dic[key][0] is numpy.ndarray:
                 problem_dic_json[key] = []
                 for element, index in enumerate(problem_dic[key]):
                     problem_dic_json[key].append(element.tolist())
         elif type(problem_dic[key]) is numpy.ndarray:
             problem_dic_json[key] = problem_dic[key].tolist()
         elif problem_dic[key] is None:
             problem_dic_json[key] = problem_dic[key]
         elif type(problem_dic[key]) is function.constant.Constant:
             problem_dic_json[key] = df.project(problem_dic[key], V).compute_vertex_values().tolist()
         elif isinstance(problem_dic[key], type(problem_dic[key])):
             if type(problem_dic[key]).__name__ in ["Electrode", "Separator"]:
                 problem_dic_2 = problem_dic[key].__dict__
                 problem_dic_json[key] = create_problem_json(problem_dic_2, V)
         else:
             pass
    return problem_dic_json

class Lagrange():
    
    def __init__(self, order, interval=[0, 1]):
        self.order = order
        self.points = numpy.linspace(interval[0], interval[1], num=order+1)
        self.f_vector()
        self.df_vector()
        self.xf_vector()
        self.xdf_vector()
    
    def simple_poly(self, point):
        poly_c = [1]
        for i in self.points:
            if i!=point:
                poly_c = polymul(poly_c, [-i/(point-i), 1/(point-i)])
        return poly_c
    
    def getPolyFromCoeffs(self, c):
        assert len(c)==self.order+1, "The length of the coefficients list has to be: "+str(self.order+1)
        poly = Polynomial([0])
        for k in range(self.order+1):
            poly = polyadd(poly, c[k]*self.f[k])
        return poly

    def f_vector(self):
        self.f = []
        for k in range(self.order+1):
            self.f.append(self.simple_poly(self.points[k]))
        
    def xf_vector(self):
        self.xf = []
        for k in range(self.order+1):
            self.xf.append(polymul([0,1],self.simple_poly(self.points[k])))

    def df_vector(self):
        self.df = []
        for k in range(self.order+1):
            self.df.append(polyder(self.simple_poly(self.points[k])))

    def xdf_vector(self):
        self.xdf = []
        for k in range(self.order+1):
            self.xdf.append(polymul([0,1],polyder(self.simple_poly(self.points[k]))))
