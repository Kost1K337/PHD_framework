from abc import ABC, abstractmethod
from src.mipt_solver.nodal.inflow_curves import Curve

class Reservoir:
    
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def update_curve(self, curve: Curve) -> Curve:
        return curve
