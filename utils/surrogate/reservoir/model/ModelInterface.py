import torch as tc
import torch.nn as nn
import numpy as np

class SModelInterface:
    def hidden_dim(self) -> int:
        pass

    def view_dim(self) -> int:
        pass

    def time_steps(self) -> int:
        pass

    def start(self) -> tc.Tensor:
        pass

    def view(self, time_step: int, hidden_cur: tc.Tensor) -> tc.Tensor:
        pass

    def veiw_weights(self, time_step: int, hidden_cur: tc.Tensor) -> tc.Tensor:
        pass

    def true_view(self, time_step: int) -> tc.Tensor:
        pass

    def step(self, time_step: int, hidden_cur: tc.Tensor) -> tc.Tensor:
        pass

    def save_hidden(self, time_step: int, hidden: tc.Tensor) -> None:
        pass

    def save_hidden_cor(self, time_step: int, hidden: tc.Tensor) -> None:
        pass

    def save_view(self, time_step: int, view: tc.Tensor) -> None:
        pass

class OptimazerInterface:
    def pre_spep(self) -> None:
        pass

    def step(self) -> None:
        pass

    def pre_epoch(self) -> None:
        pass

    def epoch(self) -> None:
        pass