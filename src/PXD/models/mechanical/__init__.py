from PXD.models.mechanical.liquid_electrolyte import LiquidElectrolyteMechanicalModel

def mechanical_model(cell):
    if cell.electrolyte.type == 'liquid':
        # raise Exception("Mechanical model for liquid electrolite cell is not implemented yet")
        return LiquidElectrolyteMechanicalModel(cell)
    elif cell.electrolyte.type in ['solid', 'polymer']:
        raise Exception("Solid electrolytes not supported in this version")

__all__ = [
    mechanical_model
]