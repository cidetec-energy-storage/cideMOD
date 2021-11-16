from .basic_mechanics import BaseMechanicalModel

class LiquidElectrolyteMechanicalModel(BaseMechanicalModel):
    def __init__(self, cell) -> None:
        pass

    def fields(self):
        return []

    def storage_order(self):
        return []

    def shape_functions(self, mesh):
        return []

    def displacement_wf(self, *args, **kwargs):
        return 0

    def hydrostatic_stress_wf(self, *args, **kwargs):
        return 0

    def fixed_bc(self, *args, **kwargs):
        return []

    def slip_bc(self, *args, **kwargs):
        return []

    def pressure_bc(self, *args, **kwargs):
        return []
        
