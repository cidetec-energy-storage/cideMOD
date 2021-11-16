import os, shutil
from dolfin import XDMFFile, MPI, Function, FunctionSpace, project, timings, TimingClear, TimingType, Timer, Constant
from array import array as pyarray
from numpy import savetxt, array, concatenate, newaxis
from sys import getsizeof
from itertools import chain


class Warehouse:
    def __init__(self, save_path, problem, delay = 1):
        self.save_path = save_path
        self.comm = MPI.comm_world
        self.problem = problem
        self.delay = delay
        self.counter = 0

        self.global_vars = {}
        self.global_var_arrays = []

    def set_delay(self, delay):
        self.delay = delay

    def internal_variables(self, fields:list):
        self.field_vars = {}
        for name in fields:
            if not isinstance(name, (list, tuple)):
                name = [name]
            if len(name) == 1:  # Store scalar
                xdmf = self._create_storing_file(self.save_path, name[0], name[0])
                fnc = self._create_storing_function(name[0])
                self.field_vars[name[0]] = (xdmf, fnc)
            elif len(name) > 1:  # Store vector or list
                valid_types = ('vector','list_of_scalar','list_of_vector')
                assert name[1] in valid_types, "Type invalid. Valid types are {}".format(valid_types)
                if name[1] == 'vector':
                    xdmf = self._create_storing_file(self.save_path, name[0], name[0])
                    fnc = self._create_storing_function(name[0],vector=True)
                    self.field_vars[name[0]] = (xdmf, fnc)
                else:
                    assert len(name) == 3, "Must specify list lenght. Format: (name[, type[, len]])"
                    assert isinstance(name[2], int), "List lenght must be an integer"
                    xdmf = [self._create_storing_file(self.save_path, name[0], '{}_{}'.format(name[0],i)) for i in range(name[2])]
                    fnc = [self._create_storing_function('{}_{}'.format(name[0],i),vector=bool(name[1]=='list_of_vector')) for i in range(name[2])]
                    self.field_vars[name[0]] = (xdmf, fnc)
            else:
                raise Exception("Internal variable only supports 3 args. Format: (name[, type[, len]])")
                
    def _create_storing_file(self, save_path, folder, name):
        xdmf = XDMFFile(self.comm ,os.path.join(save_path, folder,'{}.xdmf'.format(name)))
        xdmf.parameters['rewrite_function_mesh'] = False
        xdmf.parameters['functions_share_mesh'] = True
        xdmf.parameters['flush_output'] = True
        return xdmf

    def _create_storing_function(self, name, vector=False):
        if vector:
            fnc = Function(self.problem.V_vec, name=name)
        else:
            fnc = Function(self.problem.V, name=name)
        return fnc

    def global_variables(self, params:dict):
        self.global_var_arrays = [[] for _ in range(len(params)+1)]
        self.global_vars = params

    def post_processing_functions(self, functions:list):
        self.post_processing = functions

    def _post_process(self):
        timer = Timer('Post-processing')
        for func in self.post_processing:
            func()
        timer.stop()
        
    def _store_internals(self, time):
        timer = Timer('Store Internals')
        for name, (file, func) in self.field_vars.items():
            try:
                if 'nd_model' in self.problem.__dict__.keys():
                    variables = self.problem.nd_model.physical_variables(self.problem) 
                    var = variables[name]
                    self._store_var( var, func, file, time)
                else:    
                    index = self.problem.f_1._fields.index(name)
                    var = self.problem.f_1[index]
                    self._store_var( var, func, file, time)
            except (ValueError, KeyError) as e:
                if name in self.problem.__dict__.keys():
                    var = self.problem.__dict__[name]
                    self._store_var( var, func, file, time)
                else:
                    pass
                    # TODO: Print warning and quit name, file and func from variable lists
                    # raise Exception("Attribute '{}' not found in f_1 nor in the problem object".format(name))                
        timer.stop()

    def _store_var(self, var, func, file, time):
        if not isinstance(var,(list,tuple)):
            var = [var]
        if not isinstance(func,(list,tuple)):
            func = [func]
            file = [file]
        assert len(var) == len(func), "Specified variable length does not match"
        for (v, fnc, fout) in zip(var,func,file):
            if isinstance(var, Function):
                fout.write(fnc,time)
            else:
                if isinstance(v,(float, int)):
                    v=Constant(v)
                fnc.assign(project(v, fnc.function_space()))
                fout.write(fnc,time)

    def _store_globals(self, time):
        timer = Timer('Store Globals')
        self.global_var_arrays[0].append(time)
        for i, f in enumerate(self.global_vars.values(), 1):
            self.global_var_arrays[i].append(f['fnc']())
        timer.stop()

    def store(self, time, force = False):
        self._store_globals(time)
        if self.delay>=0:
            self.counter += 1
            if self.counter >= self.delay or force:
                self._post_process()
                self._store_internals(time)
                self.counter = 0

    def write_globals(self, clean=True):
        if MPI.rank(self.comm) == 0:
            for i, key in enumerate(self.global_vars.keys(), 1):
                global_var_array = array(self.global_var_arrays[i])
                if global_var_array.ndim == 1:
                    n = 1
                    global_var_array = global_var_array[:,newaxis]
                elif global_var_array.ndim > 2:
                    global_var_array = global_var_array.reshape(global_var_array.shape[0],global_var_array.shape[1])
                    n = global_var_array.shape[1]
                else:
                    n = global_var_array.shape[1]
                fname = os.path.join(self.save_path,'{}.txt'.format(key))
                data = concatenate((array(self.global_var_arrays[0])[:,newaxis],global_var_array), axis = 1)
                fmt = ("%2.2f \t"+ "%1.8e \t"*n)
                headers = "Time [s]\t{}".format(self.global_vars[key]['header'])
                self._save_txt_file(fname, data, fmt, headers)

            # Write condensated
            self._write_compiled_output()
            # Write timings table
            timing_table = timings(TimingClear.keep, [TimingType.wall, TimingType.user, TimingType.system])
            with open(os.path.join(self.save_path,'timings.log'), 'w') as out:
                out.write(timing_table.str(True))

            # Reset globals container
            if clean:
                self.global_variables(self.global_vars)

    def _save_txt_file(self, fname, data, fmt, headers):
        if os.path.exists(fname):
            with open(fname,'ab') as f:
                savetxt(fname=f, X=data, fmt=fmt)        
        else:
            savetxt(fname=fname, X=data, header=headers, fmt=fmt)

    def _write_compiled_output(self):
        fname = os.path.join(self.save_path,'{}.txt'.format('condensated'))
        data = []
        for var in self.global_var_arrays:
            if len(var)>0:
                if isinstance(var[0], (list, tuple)):
                    ars = array(var).T
                    for ar in ars:
                        data.append(ar)
                else:
                    data.append(array(var))
        data = array(data).T
        if data.size > 0:
            n_total = data.shape[1]-1
            data_format = ("%2.2f"+ "\t%1.8e"*n_total)
            headers = '\t'.join( ['Time [s]']+[val['header'] for key, val in self.global_vars.items()] )
            self._save_txt_file(fname, data, data_format, headers)
        
    def clean(self):
        # Close files
        for name, (file, func) in self.field_vars.items():
            file.close()
        # Write globals
        self.write_globals(True)

    def check_globals_memory_size(self, estimate = True):
        """
        Returns memory size of stored global variables in MB

        Parameters
        ----------
        estimate : bool, optional
            Wether to do a fast estimate or the slow exact measure, by default True

        Returns
        -------
        float
            memory size in MB
        """
        if estimate:
            lower_estimate = total_size([arr[0] for arr in self.global_var_arrays])
            higher_estimate = total_size([arr[-1] for arr in self.global_var_arrays])
            size = (lower_estimate + higher_estimate)*len(self.global_var_arrays)/2
            return size/2**20
        else:
            return total_size(self.global_var_arrays)

    # TODO: Method doesn't work as expected. Need to review compatibility between FEniCS and meshio.
    #       Better not use it for now. 
    # headers needed:
            # import meshio
            # from meshio.xdmf.common import xdmf_to_meshio_type
    # Needs HDF5 same version as current FEniCS version has. To have it:
    #       HDF5_VERSION=1.10.0 pip install --no-binary=h5py h5py
    #       pip install meshio
    def pack_stored_files(self):
        # Close FEniCS files
        for xdmf, _ in self.field_vars.values():
            xdmf.close()

        # Init temp containers
        pts = []
        cs = []

        ts = []
        pdata=[]
        cdata = []

        dirs = os.walk(self.save_path)
        files = []
        to_delete = []
        names = []
        for i in dirs:
            for fieldfile in i[2]:
                if fieldfile.endswith('.xdmf'):
                    files.append(os.path.join(i[0],fieldfile))
                    names.append(fieldfile.split('.')[0])
                if fieldfile.endswith('.xdmf') or fieldfile.endswith('.h5'):
                    to_delete.append(os.path.join(i[0],fieldfile))
        if len(files) == 0:
            print('No files found, unable to merge')
            
        # Aggregate output data
        for i, filename in enumerate(files):            
            with meshio.xdmf.TimeSeriesReader(filename) as reader:
                points, cells = _read_points_cells(reader)
                pts.append(points)
                cs.append(cells)
                for k in range(reader.num_steps):
                    t, point_data, cell_data = reader.read_data(k)
                    if t in ts:
                        for key, val in point_data.items():
                            pdata[k][key]=val
                        cdata[k][names[i]]=cell_data
                    else:
                        ts.append(t)
                        pdata.append({})
                        for key, val in point_data.items():
                            pdata[k][key]=val
                        cdata.append({})
                        cdata[k][names[i]]=cell_data

        # Delete output files
        for aux_file in to_delete:
            os.remove(aux_file)

        # Write merged output file
        with meshio.xdmf.TimeSeriesWriter(os.path.join(self.save_path,'internal_variables.xdmf')) as writer:
            writer.write_points_cells(points, cells)
            for i, t in enumerate(ts):
                writer.write_data(t, point_data=pdata[i])
        
        print('Files succesfully merged')

def total_size(o, handlers={}):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


# Override of mesh reader because mismatch in atributes between FEniCS and meshio
def _read_points_cells(tsreader):
    grid = tsreader.mesh_grid

    points = None
    cells = []

    for c in grid:
        if c.tag == "Topology":
            data_items = list(c)
            if len(data_items) != 1:
                raise ReadError()
            data_item = data_items[0]

            data = tsreader._read_data_item(data_item)

            # The XDMF2 key is `TopologyType`, just `Type` for XDMF3.
            # Allow both.
            if c.get("Type"):
                if c.get("TopologyType"):
                    raise ReadError()
                cell_type = c.get("Type")
            else:
                cell_type = c.get("TopologyType")

            if cell_type == "Mixed":
                cells = translate_mixed_cells(data)
            else:
                cell_type = 'Polyline' if cell_type == 'PolyLine' else cell_type
                cells.append(meshio.CellBlock(xdmf_to_meshio_type[cell_type], data))

        elif c.tag == "Geometry":
            try:
                geometry_type = c.get("GeometryType")
            except KeyError:
                pass
            else:
                if geometry_type not in ["XY", "XYZ"]:
                    raise ReadError()

            data_items = list(c)
            if len(data_items) != 1:
                raise ReadError()
            data_item = data_items[0]
            points = tsreader._read_data_item(data_item)

    tsreader.cells = cells
    return points, cells