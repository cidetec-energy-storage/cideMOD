import petsc4py

def pc_mumps(PC):
    PC.setType("lu")
    PC.setFactorSolverType("mumps")

def pc_boomerAMG(PC):
    PC.setType("hypre")
    PC.setHYPREType("boomeramg")

def pc_ffs_jacobi(PC,names, dofs):
    PC.setType("fieldsplit")
    fields = []
    for field, iset in zip(names,dofs):
        fields.append((field, petsc4py.PETSc.IS().createGeneral(iset)))
    PC.setFieldSplitIS(*fields)
    ksps = PC.getFieldSplitSubKSP()
    for ksp in ksps:
        ksp.setType("preonly")
        ksp.getPC().setType("hypre")

def pc_mixed(PC, names, dofs):
    PC.setType('composite')
    PC.setCompositeType(PC.CompositeType.MULTIPLICATIVE)
    PC.addCompositePCType('fieldsplit')
    PC.addCompositePCType('hypre')
    fs_pc = PC.getCompositePC(0)
    fields = []
    for field, iset in zip(names,dofs):
        fields.append((field, petsc4py.PETSc.IS().createGeneral(iset)))
    fs_pc.setFieldSplitIS(*fields)
    ksps = fs_pc.getFieldSplitSubKSP()
    for ksp in ksps:
        ksp.setType("preonly")
        ksp.getPC().setType("jacobi")
    fs_amg = PC.getCompositePC(1)
    fs_amg.setHYPREType("boomeramg")

