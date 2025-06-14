import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .const import * #pylint: disable=wildcard-import,unused-wildcard-import

class Pconv(nn.Module):
    """Convolution module."""
    def __init__(self, *args, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(*args, kernel_size=kernel_size, padding=0, **kwargs)
        self.ks = kernel_size

    def forward(self, x):
        x = F.pad(x, [self.ks - 1, 0])
        x = self.conv(x)
        return x

class Net(nn.Module):
    """CNN."""
    def __init__(self, i):
        super().__init__()
        self.conv = nn.Sequential(
            Pconv(2, 4, kernel_size=20, stride=1),
            nn.ELU(),
            Pconv(4, 8, kernel_size=20, stride=1),
            nn.ELU(),
            Pconv(8, i, kernel_size=20, stride=1),
        )

    def forward(self, x):
        return self.conv(x)


class LinearRelu(nn.Module):
    """Linear ReLU module."""
    def __init__(self, i):
        super().__init__()
        weights = torch.ones(i)
        # weights[:] = 1
        self._weights2 = torch.ones(1, i)
        self._weights = nn.Parameter(weights)
        self._activation = nn.ReLU()

    def forward(self, x):
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        return (x @ self._weights).reshape(*x.shape[:-1], 1)


class SaturationNet(nn.Module):
    """Network responsible for saturation fitting.s"""
    def __init__(self, n):
        super().__init__()
        self.layers = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(2, n[0]),
            nn.ELU(),
            nn.Linear(n[0], n[1]),
            nn.ELU(),
            nn.Linear(n[1], 1),
            nn.Sigmoid(),
            nn.Linear(1, 1),
        )

    def forward(self, x):
        """Forward pass"""
        return self.layers(x)


class RatesNet(nn.Module):
    """Net predicting rate."""
    def __init__(self, i, n):
        super().__init__()
        self.layers = nn.Sequential(
            LinearRelu(i), nn.Linear(1, n), nn.ELU(), nn.Linear(n, 2)
        )

    def forward(self, x):
        """Forward pass"""
        return self.layers(x)


class Simulator:
    """Rate models wrapper."""
    well_list = ["601"]

    def __init__(self, weights_path: str) -> None:
        self._pressure_models = {}
        self._initial_pressures = {}
        self._saturation_models = {}
        self._saturation_models_parameters = {}
        self._rates_models = {}
        self._rates_models_parameters = {}
        self._rs_mean = {}
        for well in self.well_list:
            with open(
                f"{weights_path}/pressure_models/initial_pressures/{well}.pickle", "rb"
            ) as f:
                self._initial_pressures[well] = pickle.load(f)
            self._pressure_models[well] = Net(self._initial_pressures[well].size)
            self._pressure_models[well].load_state_dict(
                torch.load(f"{weights_path}/pressure_models/weights/{well}.pt")
            )
            with open(
                f"{weights_path}/saturation_model/cells/{well}.pickle", "rb"
            ) as f:
                cell_list = pickle.load(f)
            self._saturation_models[well] = {}
            self._saturation_models_parameters[well] = {}
            for cell in cell_list:
                self._saturation_models[well][cell] = SaturationNet((3, 2))
                self._saturation_models[well][cell].load_state_dict(
                    torch.load(
                        f"{weights_path}/saturation_model/weights/{well}/{cell}.pt"
                    )
                )
                with open(
                    f"{weights_path}/saturation_model/parameters/{well}/{cell}.pickle",
                    "rb",
                ) as f:
                    self._saturation_models_parameters[well][cell] = pickle.load(f)
            with open(
                f"{weights_path}/rates_models/parameters/{well}.pickle", "rb"
            ) as f:
                self._rates_models_parameters[well] = pickle.load(f)
            with open(f"{weights_path}/rates_models/weights/{well}.pickle", "rb") as f:
                self._rates_models[well] = pickle.load(f)
            with open(f"{weights_path}/rates_models/rs_mean/{well}.pickle", "rb") as f:
                self._rs_mean[well] = pickle.load(f)

    def __call__(self, bhps, history, injection_total):
        res = {}
        well_dfs = dict([d for d in history.groupby(WELL_COL)])

        for well, bhp in bhps.items():
            if well_dfs.keys():
                inp = np.stack(
                    (
                        np.hstack((well_dfs[well][WBHP_COL].values.reshape(-1), [bhp])),
                        np.hstack(
                            (
                                well_dfs[well][INJT_COL].values.reshape(-1),
                                [injection_total],
                            )
                        ),
                    )
                )
                inp[0, 0] = self._initial_pressures[well].mean()
                inp = torch.diff(torch.tensor(inp.astype(float)).float())

            else:
                inp = torch.diff(
                    torch.Tensor(
                        [
                            [
                                0,
                            ],
                            [
                                0,
                            ],
                        ]
                    )
                )
            pressure = self._pressure_models[well](inp)
            pressure = (
                torch.cumsum(
                    torch.cat(
                        (
                            torch.Tensor(self._initial_pressures[well]).reshape(-1, 1),
                            pressure,
                        ),
                        dim=-1,
                    ),
                    -1,
                )
                .detach()
                .numpy()[:, -1:]
            )
            injection_cumulative = np.cumsum(
                np.hstack(
                    (well_dfs[well][INJT_COL].values.reshape(-1), [injection_total])
                )
            )[-1:]
            saturation = np.zeros(pressure.shape)
            for i, cell in enumerate(self._rates_models_parameters[well]["cells"]):
                inp = np.hstack([pressure[i], injection_cumulative])
                inp = (
                    inp
                    - self._saturation_models_parameters[well][cell]["features_mean"]
                ) / self._saturation_models_parameters[well][cell]["features_std"]
                inp = torch.Tensor(inp.astype(float)).float()
                saturation[i] = (
                    self._saturation_models[well][cell](inp).detach().numpy()
                    * self._saturation_models_parameters[well][cell]["target_std"]
                    + self._saturation_models_parameters[well][cell]["target_mean"]
                )

            krw = np.zeros(saturation.shape)
            kro = np.zeros(saturation.shape)
            parameters = self._rates_models_parameters[well]
            swn = (saturation.reshape(-1) - parameters[SWCR]) / (
                1 - parameters[SWCR] - parameters[SOWCR]
            )
            ind = np.logical_and(
                saturation.reshape(-1) >= parameters[SWL],
                saturation.reshape(-1) < parameters[SWCR],
            )
            if ind.any():
                kro[ind] = (
                    parameters[KRORW][ind]
                    + (parameters[KROLW][ind] - parameters[KRORW][ind])
                    * ((parameters[SWCR][ind] - saturation.reshape(-1)[ind]))
                ).reshape(-1, 1)
                krw[ind] = 0
            ind = np.logical_and(
                saturation.reshape(-1) > parameters[SWCR],
                saturation.reshape(-1) < 1 - parameters[SOWCR],
            )
            kro[ind] = (
                parameters[KRORW][ind] * (1 - swn[ind]) ** parameters["now"][ind]
            ).reshape(-1, 1)
            krw[ind] = (
                parameters[KRWR][ind] * swn[ind] ** parameters["nw"][ind]
            ).reshape(-1, 1)
            ind = saturation.reshape(-1) > 1 - parameters[SOWCR]
            kro[ind] = 0
            if ind.any():
                krw[ind] = (
                    parameters[KRWU][ind]
                    - (
                        (parameters[KRWU][ind] - parameters[KRWR][ind])
                        * (parameters[SWU][ind] - saturation.reshape(-1)[ind])
                        / (parameters[SOWCR][ind] + parameters[SWU][ind] - 1)
                    )
                ).reshape(-1, 1)
            depression = pressure - bhp
            n = int(self._rates_models[well]["weights"].size / 2)
            res[well] = {
                QOIL_COL: (
                    (depression + self._rates_models[well]["weights"][n:, np.newaxis])
                    * kro
                )
                .reshape(-1)
                .dot(self._rates_models[well]["weights"][:n]),
                QWAT_COL: (
                    (depression + self._rates_models[well]["weights"][n:, np.newaxis])
                    * krw
                )
                .reshape(-1)
                .dot(self._rates_models[well]["weights"][:n]),
                QGAS_COL: 0,
                PRES_COL: pressure.mean().item(),
                SAT_COL: saturation,
                KRO: kro,
                KRW: krw,
            }
            res[well][QGAS_COL] = res[well][QOIL_COL] * self._rs_mean[well]

        return res
