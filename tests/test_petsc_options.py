import petsc4py
petsc4py.init(['-help'])
from petsc4py import PETSc
# help dispatcher
COMM = PETSc.COMM_SELF

options = [
    ("snes_type", "newtonls"),
    ("ksp_type", "gmres"),
    ("pc_type", "hypre")
]

for key, value in options:
    PETSc.Options().setValue(key,value)

snes = PETSc.SNES().create(comm=COMM)
snes.setFromOptions()
snes.destroy()

# ksp = PETSc.KSP().create(comm=COMM)
# ksp.setFromOptions()
# ksp.destroy()

# pc = PETSc.PC().create(comm=COMM)
# pc.setFromOptions()
# pc.destroy()