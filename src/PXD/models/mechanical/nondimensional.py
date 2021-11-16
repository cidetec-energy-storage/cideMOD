from PXD.models.base.base_nondimensional import BaseModel

class MechanicModel(BaseModel):
    def _unscale_mechanical_variables(self, variables_dict):
        return {}

    def _scale_mechanical_variables(self, variables_dict):
        return {}
    
    def _calc_mechanic_dimensionless_parameters(self):
        pass
