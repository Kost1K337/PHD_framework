"""Functions needed for parsing json and creating the network."""
import os

from pathlib import Path

import pickle
import json
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Union

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from scipy.interpolate import UnivariateSpline
from astropy import units as u

from .. import const
from .. import custom_units as cu
from ..load_utils import load_well_data, get_vfp, get_vfp_model, load_vfp_model

from ..inflow_curves import Curve, DeclineCurveDependedOnCum, IprCurveDependedOnCum

from ..choke import SimpleChoke
from ..node import Node, NodeType
from ..pipe import Pipe, PipeType
from ..net_solver import NetSolver
from ..network import Network
from ..vfp_table import VFPTableModel, VFPWellNNModel, VFPTubeNNModel
from src.mipt_solver.nodal.metawell.model import Well_Metamodel

NODE_PIPE_MAPPING = {
    "WELL": PipeType.WELL,
    "TUBE": PipeType.TUBE,
    "CONNECTOR": PipeType.CONNECTOR,
    "JUNCTION": NodeType.JUNCTION,
    "SINK": NodeType.SINK,
    "SOURCE": NodeType.SOURCE,
}


@dataclass
class WellPI:
    name: str
    a_oil: np.ndarray
    b_oil: np.ndarray
    cum_oil: np.ndarray

    a_gas: np.ndarray
    b_gas: np.ndarray
    cum_gas: np.ndarray

    a_wat: np.ndarray
    b_wat: np.ndarray
    cum_wat: np.ndarray

    a_liq: np.ndarray
    b_liq: np.ndarray
    cum_liq: np.ndarray


def load_well_pi_from_excel(path: str) -> Dict[str, WellPI]:
    df = pd.read_excel(path, converters={'name': str})
    df.fillna(0, inplace=True)

    df_grouped = df.groupby('name')

    well_dict = {}

    for name, group in df_grouped:
        well_dict[name] = WellPI(
            name=name,
            a_oil=group['a_oil'].to_numpy() / (u.bar * u.day),
            b_oil=group['b_oil'].to_numpy() * cu.CUB_BAR,
            cum_oil=group['Cumul_oil'].to_numpy() * u.m ** 3,
            a_gas=group['a_gas'].to_numpy() / (u.bar * u.day),
            b_gas=group['b_gas'].to_numpy() * cu.CUB_BAR,
            cum_gas=group['Cumul_gas'].to_numpy() * u.m ** 3,
            a_wat=group['a_wat'].to_numpy() / (u.bar * u.day),
            b_wat=group['b_wat'].to_numpy() * cu.CUB_BAR,
            cum_wat=group['Cumul_wat'].to_numpy() * u.m ** 3,
            a_liq=group['a_liq'].to_numpy() / (u.bar * u.day),
            b_liq=group['b_liq'].to_numpy() * cu.CUB_BAR,
            cum_liq=group['Cumul_liq'].to_numpy() * u.m ** 3,
        )

    return well_dict

def preprocess_path(path: str) -> Path:
    """Parse a string with path to Path instance."""
    parts = path.split(".")
    try:
        path_str = ".".join(parts[:-1] + [parts[-1].rstrip(" /").split()[0]])
    except IndexError:
        return Path(path)
    path_str = path_str.strip(" \t\n'\"").replace("\\", "/")
    return Path(path_str)

def read_solver_config_from_json(solver: NetSolver, filepath: Union[Path, str]) -> int:
    """Read Solver Config."""
    if isinstance(filepath, Path):
        path = filepath
    if isinstance(filepath, str):
        path = preprocess_path(filepath)
    if isinstance(filepath, dict):
        solver.precision = filepath["precision"]
        solver.parse_nodal_config(filepath["nodalConfig"])
        #solver.optimization_config = filepath["optimizationConfig"]
        return 0
    if not path.is_file:
        raise FileNotFoundError("File with path '{}' not found".format(filepath))

    with open(path, "r", encoding="UTF-8") as file:
        data = json.load(file)

    if data.get("precision"):
        solver.precision = data["precision"]

    if data.get("nodalConfig"):
        solver.parse_nodal_config(data["nodalConfig"])

    if data.get("optimizationConfig"):
        solver.optimization_config = data["optimizationConfig"]

    return 0

def get_pipe_params(pipe: Pipe) -> dict:

    # diam = network['pipes'][0]['innerDiameterMm'] / 1000 # переводим в метры
    diam = pipe.d
    # roughness = network['pipes'][0]['roughnessMm'] / 1000 # переводим в метры
    roughness = pipe.r
    # tube_x0, tube_y0 = map(float, network['pipes'][0]['profileHorDistanceMSpaceHeightM'][0].split())
    tube_x0, tube_y0 = pipe.profile[0]
    tube_x1, tube_y1 = pipe.profile[1]

    tube_dx = tube_x1 - tube_x0
    tube_dy = tube_y1 - tube_y0

    measured_depth = (tube_dx ** 2 + tube_dy ** 2) ** 0.5
    angle = np.rad2deg(np.arccos(tube_dy / measured_depth))
    return {
        "measured_depth": measured_depth,
        "angle": angle,
        'roughness': roughness,
        'tube_diametr': diam,
    }
    
def get_network_params(network: Network) -> dict:
    return {
        "gas_gravity": network.rho_gas,
        "oil_gravity": network.rho_oil,
        "wat_gravity": network.rho_wat
    }


def read_network_from_json(net: Network, topology: dict) -> int:
    """Read Json and construct a Solver instance"""

    # Берем топологию из ответа бэка
    # url = service_settings.topology_url
    # url += f"{proj_id}/DownloadTopology"
    # logger.info(url)
    try:
        _nodes = topology["nodes"]
        _pipes = topology["pipes"]

        for n in _nodes:
            for k, v in list(n.items()):
                if v is None:
                    del n[k]
        topology["nodes"] = _nodes

        for p in _pipes:
            for k, v in list(p.items()):
                if v is None:
                    del p[k]
        topology["pipes"] = _pipes

    except requests.RequestException as e:
        raise e



    vfp_tables = []
    if topology.get("VFPTables"):
        net.logger.info("Loading VFP tables")
        vfp_paths = topology["VFPTables"]
        for vfp_path in vfp_paths:
            load_well_data(
                path.parent / vfp_path,
                vfp_tables,
                raise_errors=True,
                logger=net.logger,
            )

        net.logger.info("VFP tables loaded")

    if topology.get("WellsParallelCPU"):
        net.parallel = topology["WellsParallelCPU"]

    if topology.get("wellMetamodelExtension"):
        well_meta_ext = topology["wellMetamodelExtension"]
    else:
        well_meta_ext = "pickle"

    vfp_models = {}
    if topology.get("wellMetamodelsPath"):
        net.logger.info("Loading well VFP models")
#         for w_path in path.parent.glob(data["wellMetamodelsPath"]):
        for w_path in []: # TODO: задавать VFP модели сурсам Path(path).parents[1].joinpath('models', 'wells', 'weights').glob(topology["wellMetamodelsPath"]):
            well_name = w_path.stem
            net.logger.info("Loading well VFP model: {}".format(well_name))
            model = load_vfp_model(w_path)
            vfp_models[well_name] = model

    if topology.get("tubeMetamodelName"):
        tube_meta_path = topology["tubeMetamodelName"] #f'{Path(path).parents[1]}/models/tubes/{str(topology["tubeMetamodelName"])}' ### switch to full, not relative path
        # tube_meta_path = f"src/data/models/tubes/{str(data['tubeMetamodelName'])}"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.jit.load(tube_meta_path, map_location=device)
        model.eval()
        model.to(device)
        net.model = model

    if topology.get("DLLmodelPath"):
        from src.mipt_solver.nodal.network.DLLHydraulicModel.DLLHydraulicModel import DLLHydraulicModel #### FIXME: change to real DLL model inference
        net.model = DLLHydraulicModel(path = topology["DLLmodelPath"]) 

    if topology.get("tubeMetamodelFeatureDescr"):
#         tube_meta_descr_path = path.parent / str(data["tubeMetamodelFeatureDescr"])
        tube_meta_descr_path = topology["tubeMetamodelFeatureDescr"]# f'{Path(path).parents[1]}/models/tubes/{str(topology["tubeMetamodelFeatureDescr"])}' ### switch to full, not relative path
        # tube_meta_descr_path = f"src/data/models/tubes/{str(data['tubeMetamodelFeatureDescr'])}"
        with open(tube_meta_descr_path, "r", encoding="utf-8") as f:
            feature_descr = json.load(f)
        tube_metamodel_scaler = StandardScaler()
        tube_mm_features = ['GOR', 'LIQ', 'THP', 'WCT', 'angle', 'friction', 'gravity', 'measured_depth', 'tube_diametr']
        tube_metamodel_scaler.scale_ = np.array([feature_descr["std"][k] for k in tube_mm_features])
        tube_metamodel_scaler.mean_ = np.array([feature_descr["mean"][k] for k in tube_mm_features])
        std_scale = (feature_descr["log_pressure_drop_mean"], feature_descr["log_pressure_drop_std"])
        
    if topology.get("minMaxScalerPath"):
        scaler_path = Path(topology.get("minMaxScalerPath"))
        min_values = np.loadtxt(scaler_path / 'min_values.txt')
        max_values = np.loadtxt(scaler_path / 'max_values.txt')
        scaler = MinMaxScaler()
        scaler.fit([min_values[1:], max_values[1:]])
        tube_metamodel_scaler = scaler
        net.scale = (min_values[0], max_values[0])

    if topology.get("tubeScalerName"):
        tube_scaler_path = path.parent / str(topology["tubeScalerName"])
        with open(tube_scaler_path, "rb") as file:
            net.scaler = pickle.load(file)

    if topology.get("gasDensityKgToM3"):
        net.rho_gas = topology["gasDensityKgToM3"] * cu.KG_M3

    if topology.get("VFPIPRMethod"):
        temp = topology["VFPIPRMethod"]
        if temp not in ("minimize", "cobweb", "bisect"):
            raise ValueError(
                "Only these methods are allowed: {variants}. Given: {given}".format(
                    variants=("minimize", "cobweb", "bisect"), given=temp
                )
            )
        net.vfp_ipr_method = topology["VFPIPRMethod"]

    if topology.get("oilDensityKgToM3"):
        net.rho_oil = topology["oilDensityKgToM3"] * cu.KG_M3

    if topology.get("waterDensityKgToM3"):
        net.rho_wat = topology["waterDensityKgToM3"] * cu.KG_M3

    if topology.get("globalNodesCount"):
        net.num_nodes = topology.get("globalNodesCount")
        for uid in range(topology["globalNodesCount"]):
            net.nodes.append(Node(id=uid))

    if topology.get("globalPipesCount"):
        net.num_pipes = topology.get("globalPipesCount")
        for uid in range(topology["globalPipesCount"]):
            net.pipes.append(Pipe(id=uid))

    pi_by_well_dict: Dict[str, WellPI] = {}

    #if topology.get("piCurveExcelPath"):
        #pi_by_well_dict = load_well_pi_from_excel(topology.get("piCurveExcelPath")) #f'{Path(path).parents[1]}/historical_data/{topology.get("piCurveExcelPath")}') ### switch to full, not relative path

    if topology.get("nodes"):
        node_dicts = topology["nodes"]
        for node_dict in node_dicts:
            if node_dict.get("id") is None:
                return 2

            node_id = node_dict["id"]

            if node_id < 0 or node_id >= len(net.nodes):
                return 3

            node = net.nodes[node_id]
            oil_debit = None
            wat_debit = None
            gas_debit = None
            liq_rate_cubes = None
            k_gas = None
            k_liquid = None
            p_reservoir = None

            for attr, value in node_dict.items():
                if attr == "id":
                    node.id = int(value)
                elif attr == "type":
                    node.ntype = NODE_PIPE_MAPPING.get(value, None)
                elif attr == "name":
                    node.name = value
                elif attr == "blackOilPVTModelID":
                    node.black_oil_pvt_model_id = value
                elif attr in ("inletPressureAtma", "inletPressureAtm"):
                    node.p = (value * cu.ATM).to(u.Pa)
                elif attr == "inletPressureBar":
                    node.p = (value * u.bar).to(u.Pa)
                elif attr in "tableGasDebit":
                    list_bhp = value["BHP"]
                    list_debit = value["Q"]
                    node.debit_gas_function = UnivariateSpline(
                        list_bhp, list_debit, k=1
                    )
                elif attr == "tableOilDebit":
                    list_bhp = value["BHP"]
                    list_debit = value["Q"]
                    node.debit_oil_function = UnivariateSpline(
                        list_bhp, list_debit, k=1
                    )
                elif attr == "tableWatDebit":
                    list_bhp = value["BHP"]
                    list_debit = value["Q"]
                    node.debit_water_function = UnivariateSpline(
                        list_bhp, list_debit, k=1
                    )
                elif attr == "GOR":
                    node.gor = value * cu.M3_M3
                elif attr == "WCT":
                    node.wct = value * cu.M3_M3
                #elif attr == "ALQ": # NOTE: Pipe property
                    #node.alq = value
                elif attr in ("liqRateM3ToDay", "liqRateM3ToD"):
                    # m^3 / day
                    liq_rate_cubes = value * cu.CUB
                elif attr in ("liqRateLimitM3ToDay", "liqRateLimitM3ToD"):
                    # m^3 / day
                    if len(value) == 2:
                        node.q_b = value[0] * cu.CUB, value[1] * cu.CUB
                    elif len(value) == 1:
                        node.q_b = 0 * cu.CUB, value[0] * cu.CUB
                elif attr in ("oilRateTonneToDay", "oilRateTonneToD"):
                    oil_debit = value * cu.TON_DAY
                elif attr in ("waterRateTonneToDay", "waterRateTonneToD"):
                    wat_debit = value * cu.TON_DAY
                elif attr in ("gasRateTonneToDay", "gasRateTonneToD"):
                    gas_debit = value * cu.TON_DAY
                elif attr in ("debitLiquidCoefficientM3ToDayToBar", "debitLiquidCoefficientM3ToDToBar"):
                    # m^3 / (day * bar) -> m^3 / (day * atm)
                    k_liquid = (value * cu.CUB_BAR).to(cu.CUB_ATM)
                elif attr in ("debitGasCoefficientM3ToDayToBar", "debitGasCoefficientM3ToDToBar"):
                    k_gas = (value * cu.CUB_BAR).to(cu.CUB_ATM)
                elif attr in ("reservoirPressureAtma", "reservoirPressureAtm"):
                    p_reservoir = value * cu.ATM
                elif attr == "reservoirPressureBar":
                    p_reservoir = (value * u.bar).to(cu.ATM)

            node.k_liquid = k_liquid
            node.p_reservoir = p_reservoir
            node.k_gas = k_gas

            # Это инициализация кривых из джсона, предположение не верно, можно не читать))
            # ? возможно на первом расчетном шаге расчитываются параметры из условия, после чего условие не выполняется и кривые не пересчитываются что дает одинаковым значения в чем и есть проблема
            if k_liquid is not None and p_reservoir is not None:

                # Инициализация IPR кривой
                # Есть два цикла: по временным шагам и по итерациям внутри шага для схождения сетки
                # Кривые инициализируются один раз на временной шаг, потом не изменяются
                # Pr и J должны быть от модели притока на каждом временном шаге
                params = const.DEFAULT_IPR_DICT.copy()
                params["Pr"] = p_reservoir.to(u.bar) # P пластовое
                params["J"] = k_liquid.to(cu.CUB_BAR) # коэф продуктивности
                curve = Curve(params, "ipr")
                node.ipr_curve = curve

            if k_gas is not None:
                params = const.DEFAULT_GAS_DICT.copy()
                params["Pr"] = p_reservoir.to(u.bar)
                params["J"] = k_gas.to(cu.CUB_BAR)
                curve = Curve(params, "ipr")
                node.ipr_curve = curve

            if not any(
                v is None
                for v in (
                    liq_rate_cubes,
                    node.gor,
                    node.wct,
                    net.rho_gas,
                    net.rho_wat,
                    net.rho_oil,
                )
            ):
                oil_debit = (liq_rate_cubes * (1 - node.wct) * net.rho_oil).to(
                    cu.TON_DAY
                )
                wat_debit = (liq_rate_cubes * node.wct * net.rho_wat).to(cu.TON_DAY)
                gas_debit = (
                    liq_rate_cubes * (1 - node.wct) * node.gor * net.rho_gas
                ).to(cu.TON_DAY)

            if not any(
                v is None
                for v in (
                    oil_debit,
                    wat_debit,
                    gas_debit,
                )
            ):
                debit = oil_debit + wat_debit + gas_debit
                node.q = debit.to(cu.KG_SEC)  # ton/d -> kg/sec
                node.mfo = oil_debit / debit
                node.mfw = wat_debit / debit
                gor = (gas_debit * net.rho_oil) / (oil_debit * net.rho_gas)
                if node.gor is None or np.isnan(node.gor.value).any():
                    node.gor = gor
                wct = (wat_debit / net.rho_wat) / (oil_debit / net.rho_oil + wat_debit / net.rho_wat)
                if node.wct is None or np.isnan(node.wct.value).any():
                    node.wct = wct
            """
            if node.name in pi_by_well_dict:
                if node_dict.get('reservoirPressureAtma'):
                    reservoir_pressure = node_dict.get('reservoirPressureAtma') * cu.ATM
                elif node_dict.get('reservoirPressureBar'):
                    reservoir_pressure = node_dict.get('reservoirPressureBar') * u.bar
                else:
                    raise ValueError(f'Reservoir pressure is not set for node {node_id}')

                # Старый костыль, без ML модели
                ipr_curve = IprCurveDependedOnCum(
                    Pbhp_prev=reservoir_pressure,
                    cum_ab_table={
                        'a': pi_by_well_dict[node.name].a_liq,
                        'b': pi_by_well_dict[node.name].b_liq,
                        'cum': pi_by_well_dict[node.name].cum_liq,
                    }
                )

                wat_curve = DeclineCurveDependedOnCum(
                    kind='wat',
                    Pbhp_prev=reservoir_pressure,
                    cum_ab_table={
                        'a': pi_by_well_dict[node.name].a_wat,
                        'b': pi_by_well_dict[node.name].b_wat,
                        'cum': pi_by_well_dict[node.name].cum_wat,
                    }
                )

                oil_curve = DeclineCurveDependedOnCum(
                    kind='oil',
                    Pbhp_prev=reservoir_pressure,
                    cum_ab_table={
                        'a': pi_by_well_dict[node.name].a_oil,
                        'b': pi_by_well_dict[node.name].b_oil,
                        'cum': pi_by_well_dict[node.name].cum_oil,
                    }
                )
                
                gas_curve = DeclineCurveDependedOnCum(
                    kind='gas',
                    Pbhp_prev=reservoir_pressure,
                    cum_ab_table={
                        'a': pi_by_well_dict[node.name].a_gas,
                        'b': pi_by_well_dict[node.name].b_gas,
                        'cum': pi_by_well_dict[node.name].cum_gas,
                    }
                )

                node.set_curve(ipr_curve, 'ipr')
                node.set_curve(wat_curve, 'wat')
                node.set_curve(oil_curve, 'oil')
                node.set_curve(gas_curve, 'gas')
            """
    if topology.get("pipes"):
        pipe_dicts = topology["pipes"]
        num_chokes = 0
        for pipe_dict in pipe_dicts:
            if pipe_dict.get("id") is None:
                return 2

            pipe_id = pipe_dict["id"]

            if pipe_id < 0 or pipe_id >= len(net.pipes):
                return 3
            pipe = net.pipes[pipe_id]

            for attr, value in pipe_dict.items():
                if attr == "id":
                    pipe.id = int(value)
                elif attr == "type":
                    pipe.ptype = NODE_PIPE_MAPPING.get(value, None)
                elif attr == "name":
                    pipe.name = value
                elif attr == "inletNodeId":
                    pipe.fnode[0] = value
                elif attr == "outletNodeId":
                    pipe.fnode[1] = value
                elif attr == "innerDiameterMm":
                    pipe.d = (value * u.mm).to(u.m)
                elif attr == "roughnessMm":
                    pipe.r = (value * u.mm).to(u.m)
                elif attr == "alq":
                    pipe.alq = value  
                elif attr == "wellMetamodelName":
                    model = torch.jit.load(value)
                    model.eval()
                    pipe.model = model
                elif attr == "wellScalerName":
                    scaler = pickle.load(open(value, "rb"))
                    pipe.scaler = scaler
                elif attr == "VFPNumber":
                    pipe.vfp_model = VFPTableModel(
                        get_vfp(vfp_tables, num=value)
                    )
                elif attr == "VFPModelNumber":
                    pipe.vfp_model = VFPWellNNModel(
                        get_vfp_model(vfp_models, name=value)
                    )
                elif attr == "Well_Metamodel":
                    pipe.vfp_model = Well_Metamodel.load(file_path=value)                    


                elif attr == "DLLTubeModel":
                    pipe.vfp_model = DLLHydraulicModel(path = topology["DLLmodelPath"])

                elif attr == "VFPTubeModel":
                    pipe.vfp_model = VFPTubeNNModel(
                        model=net.model,
                        scaler=tube_metamodel_scaler,
                        rescale=std_scale,
                        net_params=get_network_params(net),
                        pipe_params=get_pipe_params(pipe),
                        **value
                    )
                elif attr == "choke":
                    choke_dct = value.copy()

                    if choke_dct.get("diffPres"):
                        choke_dct["diffPres"] = (choke_dct["diffPres"] * u.bar).to(
                            cu.ATM
                        )
                    choke_dct["id"] = num_chokes
                    num_chokes += 1
                    pipe.choke = SimpleChoke(choke_dct)
                elif attr == "profileHorDistanceMSpaceHeightM":
                    for prof in value:
                        horlength, height = prof.split()
                        horlength = float(horlength) * u.m
                        height = float(height) * u.m
                        pipe.profile.append((horlength, height))

    # Изменения по температурной модели (добавление свойств PVT и температур)
    if topology.get("surroundingTemperatureDegC"):
        net.surroundingTemperatureDegC = topology.get("surroundingTemperatureDegC") * cu.DEG_CELS

    if topology.get("reservoirTemperatureDegC"):
        net.reservoirTemperatureDegC = topology.get("reservoirTemperatureDegC") * cu.DEG_CELS

    if topology.get("heatTransferCoefficientWToM2ToDegK"):
        net.heatTransferCoefficientWToM2ToDegK = topology.get("heatTransferCoefficientWToM2ToDegK") * cu.W_M2_DEG_KELV

    if topology.get("oilHeatCapacityJToKgToDegK"):
        net.oilHeatCapacityJToKgToDegK = topology.get("oilHeatCapacityJToKgToDegK") * cu.J_KG_DEG_K

    if topology.get("watHeatCapacityJToKgToDegK"):
        net.watHeatCapacityJToKgToDegK = topology.get("watHeatCapacityJToKgToDegK") * cu.J_KG_DEG_K

    if topology.get("gasHeatCapacityJToKgToDegK"):
        net.gasHeatCapacityJToKgToDegK = topology.get("gasHeatCapacityJToKgToDegK") * cu.J_KG_DEG_K

    return 0
