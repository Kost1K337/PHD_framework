
"""Solver class."""
import warnings
import logging
from copy import deepcopy
import traceback

from typing import Any, Dict, Iterable, List, Tuple, Union
import numpy as np
import multiprocessing as mp
from multiprocessing.connection import Connection

from astropy import units as u
from ..well_utils import VFP_IPR_METHODS

from .. import custom_units as cu

from ..const import *  # pylint: disable=wildcard-import,unused-wildcard-import
from .network_defaults import *  # pylint: disable=wildcard-import,unused-wildcard-import
from ..df_utils import get_network_table

from ..inflow_curves.curve import Curve
from ..choke import Choke
from ..node import Node, NodeType
from ..pipe import Pipe, PipeType

from ..pvt_model import pvt_model

from ...nodal.network.DLLHydraulicModel.DLLHydraulicModel import DLLHydraulicModel

import math
import pprint as pp

NODE_PIPE_MAPPING = {
    "WELL": PipeType.WELL,
    "TUBE": PipeType.TUBE,
    "CONNECTOR": PipeType.CONNECTOR,
    "JUNCTION": NodeType.JUNCTION,
    "SINK": NodeType.SINK,
    "SOURCE": NodeType.SOURCE,
}

import numpy as np

def get_neighbour_id(pipe_nodes_ids, node_id):
    return pipe_nodes_ids[1] if pipe_nodes_ids[0] == node_id else pipe_nodes_ids[0]


def checkSPD(Y):
    assert np.all((Y.T - Y) == 0), "Matrix `Y` is not symmetric!"
    try:
        np.linalg.cholesky(Y)
    except np.linalg.LinAlgError as e:
        raise AssertionError("Matrix `Y` is not positive-definite!") from e


def create_well_process(nodes: List[Node], pipe: Pipe, ws: Connection, damping: float, densities: Dict[str, Any]) -> None:
    while True:
        msg = ws.recv()
        if msg[0] == "run":
            n_0 = nodes[pipe.fnode[0]]
            n_1 = nodes[pipe.fnode[1]]
            n_0.p = msg[1]
            n_1.p = msg[2]
            result = calculate_well_resistance(nodes, pipe, densities)
            prev_res = pipe.res
            curr_res = result[0]
            pipe.res = prev_res + damping * (curr_res - prev_res)
            ws.send(result)

def calculate_well_resistance(nodes: List[Node], pipe: Pipe, densities: Dict[str, Any]) -> u.Quantity:
    d_h = 0.0 * u.m
    n_0 = nodes[pipe.fnode[0]]
    n_1 = nodes[pipe.fnode[1]]
    try:
        if n_0.ntype == NodeType.SOURCE:
            q_oil_mass, q_wat_mass, q_mass = calculate_well_rate(n_0, n_1, pipe, densities)
            bhp, thp = n_0.p_a, n_1.p_a
        elif n_1.ntype == NodeType.SOURCE:
            q_oil_mass, q_wat_mass, q_mass = calculate_well_rate(n_1, n_0, pipe, densities)
            thp, bhp = n_0.p_a, n_1.p_a

        res = abs((bhp - thp) / pipe.q).to(cu.PA_KG_S)
        res = res if d_h < 0.0 else abs(res)
        result = res, {"bhp": bhp, "thp": thp, "q_oil_mass": q_oil_mass, 
                       "q_wat_mass": q_wat_mass, "q_mass": q_mass}
        return result
    except Exception as e:
        raise RuntimeError(
            f"Error while calculating tube resistance, pipe id: {pipe.id}"
        ) from e

def calculate_well_rate(bh_node: Node, 
                        th_node: Node, 
                        pipe: Pipe,
                        densities: Dict[str, Any]
    ) -> None:
    """Calculate and update well rate parameters."""
    q_gas, q_liq = 0.0, 0.0

    gor = bh_node.gor
    wct = bh_node.wct
    bhp = bh_node.p_a
    alq = bh_node.alq  # artificial lift quantity
    
    kwargs = densities
    kwargs["alq"] = alq

    vfp_ipr_solver = VFP_IPR_METHODS["minimize"]
    q_liq, bhp = vfp_ipr_solver(bh_node, th_node, pipe, **kwargs)
    gas_kwargs = {"gor": gor, "wct": wct, "qliq": q_liq}
    q_gas = bh_node.get_qgas(bhp, **gas_kwargs)
    
    ## mass flow rates
    q_wat_mass = (densities["rho_wat"] * wct * q_liq).to(cu.KG_SEC)  # kg/s
    q_oil_mass = (densities["rho_oil"] * (1 - wct) * q_liq).to(cu.KG_SEC)  # kg/s
    q_gas_mass = (densities["rho_gas"] * q_gas).to(cu.KG_SEC)  # kg/s
    q_mass = q_wat_mass + q_oil_mass + q_gas_mass  # kg/s
    
    ## Update borehole node params
    bh_node.p_a = bhp
    bh_node.q = q_mass
    bh_node.mfw = q_wat_mass / q_mass
    bh_node.mfo = q_oil_mass / q_mass

    ## Update well pipe params
    pipe.q = bh_node.q
    pipe.mfo = bh_node.mfo
    pipe.mfw = bh_node.mfw   
    return q_oil_mass, q_wat_mass, q_mass

def update_well_state(nodes: List[Node], pipe: Pipe, **kwargs) -> u.Quantity:
    n_0 = nodes[pipe.fnode[0]]
    n_1 = nodes[pipe.fnode[1]]
    try:
        if n_0.ntype == NodeType.SOURCE:
            bh_node = n_0
        elif n_1.ntype == NodeType.SOURCE:
            bh_node = n_1

        ## Update borehole node params
        bh_node.p_a = kwargs["bhp"]
        bh_node.q = kwargs["q_mass"]
        bh_node.mfw = kwargs["q_wat_mass"] / kwargs["q_mass"]
        bh_node.mfo = kwargs["q_oil_mass"] / kwargs["q_mass"]

        ## Update well pipe params
        pipe.q = kwargs["q_mass"]
        pipe.mfo = bh_node.mfo
        pipe.mfw = bh_node.mfw   

    except Exception as e:
        raise RuntimeError(
            f"Error while updating well properties, pipe id: {pipe.id}"
        ) from e


class PipeResistanceCalculationError(Exception):
    """Raise when there is error while calculating pipe resistance"""


class DefaultValueWarning(Warning):
    """Printed when no values are provided and defaults are used when not recommended."""


class InvalidTopologyError(Exception):
    """Raised when a topology is invalid"""


class NetworkParameterNotSetError(Exception):
    """Raised when a parameter in network is not set."""


class VFPInputOutOfRangeError(Exception):
    """Raised when an input parameter in VFP calculation is out of range."""
    pass

class Network:
    """Network object."""

    def __init__(self):
        """Init Network."""
        self.senders: Dict[str, Any] = {}
        self.count_iters: int = 0
        # hydrodynamic resistance tolerance
        self.precision: float = TOL_RES
        # standard cond gas density kg/m3
        self.rho_gas_standart: u.Quantity = RHO_GAS * cu.KG_M3
        self.rho_gas = self.rho_gas_standart
        # standard cond water density kg/m3
        self.rho_wat_standart: u.Quantity = RHO_WAT * cu.KG_M3
        self.rho_wat = self.rho_wat_standart
        # standard cond oil density kg/m3
        self.rho_oil_standart: u.Quantity = RHO_OIL * cu.KG_M3
        self.rho_oil = self.rho_oil_standart
        self.m_flagtask: bool = False
        self.m_flagmomw: bool = True
        self.nodes: List[Node] = []
        self.num_nodes: int = 0
        self.pipes: List[Pipe] = []
        self.num_pipes: int = 0
        self.m_flagmix: bool = False
        # Average (in inverse flow constant) mixture oil fraction
        self.m_mfo: float = 0.0
        # Average (in inverse flow constant) mixture water fraction
        self.m_mfw: float = 0.0
    
        # Для реализации температурной модели
        self.surroundingTemperatureDegC = 0 * cu.DEG_CELS # Температура окружающей среды
        self.reservoirTemperatureDegC = 0 * cu.DEG_CELS # Пластовая температура (от нее начинается расчет температуры в сети)
        self.heatTransferCoefficientWToM2ToDegK = 0 * cu.W_M2_DEG_KELV # Коэффициент теплопроводности общий для модели
        self.oilHeatCapacityJToKgToDegK = 0 * cu.J_KG_DEG_K # Теплоемкость нефти
        self.watHeatCapacityJToKgToDegK = 0 * cu.J_KG_DEG_K # Теплоемкость воды
        self.gasHeatCapacityJToKgToDegK = 0 * cu.J_KG_DEG_K # Теплоемкость газа
        self.node_topology = dict() # Хранит топологию сети для рекурсивного обхода
        self.pipe_topology = dict() # Хранит топологию сети для рекурсивного обхода

        self.model: Any = None
        self.scaler: Any = None
        self.scale: tuple = (0,1)

        self.ipr_dict: Dict[str, Any] = {}
        self.wct_dict: Dict[str, Any] = {}
        self.gor_dict: Dict[str, Any] = {}
        self._autochokes: List[Choke] = []
        self._wellnames: List[str] = []
        self._wellnodes: List[Node] = []
        self._sink: Node = None
        self.non_fixed_nodes: List[Node] = []

        self.logger: logging.Logger = logging.getLogger("Network")
        self.thp_dict_num: Dict[int, int] = {}
        self.vfp_ipr_method: str = "minimize"
        self.is_parallel: bool = False

        # relative densities
        self.gamma_gas_standart: u.Quantity = RHO_GAS / RHO_AIR_REL * cu.KG_M3_KG_M3
        self.gamma_oil_standart: u.Quantity = RHO_OIL / RHO_WAT_REL * cu.KG_M3_KG_M3
        self.gamma_wat_standart: u.Quantity = RHO_WAT / RHO_WAT_REL * cu.KG_M3_KG_M3

        self.gamma_gas = self.gamma_gas_standart
        self.gamma_oil = self.gamma_oil_standart
        self.gamma_wat = self.gamma_wat_standart

        self.pvt_model = pvt_model(gamma_gas=self.gamma_gas, gamma_oil=self.gamma_oil, gamma_wat=self.gamma_wat, phase_ratio={'type': 'rsb', 'value': 0})

        # TODO: инкапсулировать все pvt свойства в один атрибут 
        self.pb = self.pvt_model.pb
        self.rs = self.pvt_model.rs
        self.muo = self.pvt_model.muo
        self.mul = self.pvt_model.mul
        self.bo = self.pvt_model.bo
        self.compro = self.pvt_model.compro
        self.z = self.pvt_model.z
        self.bg = self.pvt_model.bg
        self.mug = self.pvt_model.mug
        self.bw = self.pvt_model.bw
        self.comprw = self.pvt_model.comprw
        self.muw = self.pvt_model.muw
        self.salinity = self.pvt_model.salinity
        self.st_wat_gas = self.pvt_model.st_wat_gas
        self.st_oil_gas = self.pvt_model.st_oil_gas
        self.st_liq_gas = self.pvt_model.st_liq_gas
        self.hc_wat = self.pvt_model.hc_wat
        self.hc_gas = self.pvt_model.hc_gas
        self.hc_oil = self.pvt_model.hc_oil
        self.mum = self.pvt_model.mum
        self.comprg = self.pvt_model.comprg

    @property
    def pvt_properties(self):
        """Возвращает словарь с текущими значениями всех атрибутов"""
        return {
            'pb': self.pb,
            'rs': self.rs,
            'muo': self.muo,
            'mul': self.mul,
            'bo': self.bo,
            'compro': self.compro,
            'z': self.z,
            'bg': self.bg,
            'mug': self.mug,
            'bw': self.bw,
            'comprw': self.comprw,
            'muw': self.muw,
            'salinity': self.salinity,
            'st_wat_gas': self.st_wat_gas,
            'st_oil_gas': self.st_oil_gas,
            'st_liq_gas': self.st_liq_gas,
            'hc_wat': self.hc_wat,
            'hc_gas': self.hc_gas,
            'hc_oil': self.hc_oil,
            'mum': self.mum,
            'comprg': self.comprg
        }        

    def __str__(self):
        return f"Network:\nNodes: {self.nodes}\n{self.pipes}"

    @property
    def autochokes(self):
        self._autochokes = [
            pipe.choke for pipe in self.pipes if pipe.choke and pipe.choke.is_auto
        ]
        return self._autochokes

    @property
    def wellnames(self):
        self._wellnames = [n.name for n in self.nodes if n.ntype == NodeType.SOURCE]
        return self._wellnames

    @property
    def wellnodes(self):
        self._wellnodes = [n for n in self.nodes if n.ntype == NodeType.SOURCE]
        return self._wellnodes

    def _default_ipr_inv_function(self, P, **kwargs):
        """Linear IPR function."""
        return kwargs["J"] * (kwargs["Pr"] - P)

    @property
    def df(self):
        return get_network_table(self)

    @property
    def sink(self):
        """Sink node"""
        for node in self.nodes:
            if node.ntype == NodeType.SINK:
                self._sink = node
                return self._sink
        raise ValueError(f"Sink hasn't been found.")

    def select_nodes(self, criterion=lambda n: n, mapping=lambda x: x):
        """Return all nodes which satisfy the criterion."""
        return [mapping(n) for n in filter(criterion, self.nodes)]

    def set_curves(self, curves: Dict[str, Curve], kind: str): 
        for well in self.wellnodes:
            if well.name in curves:
                well.set_curve(curves[well.name], kind)

    def set_curves_from_dict(self, dct: Union[Dict[str, Any], None], kind: str = "ipr") -> None:
        """Method to set curves."""
        if not dct or kind not in ("ipr", "wct", "gor"):
            return

        dct_ = deepcopy(dct)
        setattr(self, kind + "_dict", dct_)

        for well in self.wellnodes:
            try:
                params = dct_[well.name]
            except KeyError:
                warnings.warn(
                    "Well not found: {}. Default {} parameters are used".format(
                        well.name, kind
                    ),
                    DefaultValueWarning,
                )
                params = DEFAULT_CURVE_DICT[kind].copy()

            curve = Curve(params, kind)
            well.set_curve(curve, kind)

    def get_adjacency_dict(self) -> Dict[int, List[int]]:
        """Get adjacency dict."""
        adj_dict = {}

        for inode in range(self.num_nodes):
            current_node = self.nodes[inode]
            adj_nodes = []
            for ilink in range(current_node.nlink):
                ipipe = current_node.link[ilink]
                pipe = self.pipes[ipipe]
                ineighbour = pipe.fnode[1] if pipe.fnode[0] == inode else pipe.fnode[0]
                adj_nodes.append(ineighbour)
            adj_dict[inode] = adj_nodes

        return adj_dict

    def dfs(
        self, start: int, adj_dict: Dict[int, List[int]], reached_nodes: List[int]
    ) -> None:
        """Depth-first search."""
        reached_nodes.add(start)

        values = adj_dict.get(start)

        if values is not None:
            for value in values:
                if value not in reached_nodes:
                    self.dfs(value, adj_dict, reached_nodes)

    def check_topology(self) -> None:
        """Check topology connectedness by checking reached nodes."""
        adj_dict = self.get_adjacency_dict()
        reached = set()
        self.dfs(0, adj_dict, reached)
        num_reached = len(reached)
        unreached = set(range(self.num_nodes)).difference(reached)

        if num_reached != self.num_nodes:
            raise InvalidTopologyError(
                "Topology is not connected!"
                + f"Number of reached nodes != number of all nodes: {num_reached} != {self.num_nodes}. "
                + f"List of reached nodes: {reached}. "
                + f"List of unreached nodes: {unreached}."
            )

        self.logger.info("Topology checked!")


    def define_new_topology_structure(self) -> tuple:
        """
        Функция реализована 16.12.2024. Поскольку не было найдено функционала по поиску соседей для объектов.
        Создается структуры для обеспечения воозможности рекурсивного обхода сети
        """

        node_topology = {i.id: {'conns': ([], []), 'obj': i} for i in self.nodes}
        pipe_topology = {}

        for i in self.pipes:
            links = i.fnode

            node_topology[links[0]]['conns'][0].append(links[1])
            node_topology[links[1]]['conns'][1].append(links[0])

            pipe_topology[(links[0], links[1])] = i

        return (node_topology, pipe_topology)


    def _calc_source_heat_in(self, node: Node)->tuple:
        """
        Возвращает тепловой поток (температура в Кельвинах, но возвращается в C)
        На входе node.ntype = NodeType.SOURCE # Только для источников вызывается эта функция

        return:
        q - mass flow, kg/s
        t_in - degrees Celsius
        q_heat_in - Watt
        """
        q = node.q
        t_in = self.reservoirTemperatureDegC
        q_oil_heat_in = node.mfo * self.oilHeatCapacityJToKgToDegK
        q_wat_heat_in = node.mfw * self.watHeatCapacityJToKgToDegK
        q_gas_heat_in = node.mfg * self.gasHeatCapacityJToKgToDegK
        q_heat_body = (q_oil_heat_in + q_wat_heat_in + q_gas_heat_in) * node.q
        q_heat_in = q_heat_body * t_in.to(cu.DEG_KELV, u.temperature())
        return (q, node.mfo, node.mfw, t_in, q_heat_in)

    def precalc_network_heat_rates(self):
        """
        Функция выполняет рекурсивный обход сети, начиная от источников рассчитывает температуры
        Также переопределяет дебиты и mfo, mfw
        """

        def _calc_heat_in_from_inlet_nodes(node: Node, nodes_flag_dict: dict) -> tuple:
            """
            Рассчитывается смешение потоков.

            return:
            q - mass flow, kg/s
            mfo - массовая доля нефти в потоке, -
            mfw - массовая доля воды в потоке, -
            t_in - degrees Celsius
            q_heat_in - Watt
            """

            node_id = node.id

            prev_q_list = []
            prev_q_oil_list = []
            prev_q_wat_list = []
            prev_q_heat_out_list = []
            prev_q_heat_body_list = []
            prev_t_out_list = []
            inlet_nodes_id = self.node_topology[node_id]['conns'][0]
            for sub_node_id in inlet_nodes_id:
                sub_node = self.node_topology[sub_node_id]['obj']

                # Узел - СТОК не попадет в рекурсивный обход
                if nodes_flag_dict[sub_node_id]:
                    pass
                else:
                    up_recursive_fun(sub_node, nodes_flag_dict)

                pipe_obj = self.pipe_topology[(node_id, sub_node_id)]

                prev_q_list.append(pipe_obj.q)
                prev_q_oil_list.append(pipe_obj.q*pipe_obj.mfo)
                prev_q_wat_list.append(pipe_obj.q*pipe_obj.mfw)
                prev_q_heat_out_list.append(pipe_obj.q_heat_out)
                prev_q_heat_body_list.append(pipe_obj.q_heat_out / pipe_obj.t_out.to(cu.DEG_KELV, u.temperature()))
                prev_t_out_list.append(pipe_obj.t_out)

            q = sum(prev_q_list)
            mfo = sum(prev_q_oil_list)/q
            mfw = sum(prev_q_wat_list)/q
            q_heat_in = sum(prev_q_heat_out_list)
            t_in = sum([i/q*j for i, j in zip(prev_q_list, prev_t_out_list)])

            return (q, mfo, mfw, t_in, q_heat_in)

        def up_recursive_fun(node, nodes_flag_dict:dict):
            """
            Рекурсивный обход
            """

            node_id = node.id

            if node.ntype == NodeType.SINK:
                q, mfo, mfw, t_in, q_heat_in = _calc_heat_in_from_inlet_nodes(node, nodes_flag_dict)

                node.q = q
                node.mfo = mfo
                node.mfw = mfw
                node.t_in = t_in
                node.t_out = t_in
                node.q_heat_in = q_heat_in
                node.q_heat_out = q_heat_in
                nodes_flag_dict[node_id] = 1
                return nodes_flag_dict

            # В случае, если узел является 
            if node.ntype == NodeType.SOURCE:
                q, mfo, mfw, t_in, q_heat_in = self._calc_source_heat_in(node)
            else:
                q, mfo, mfw, t_in, q_heat_in = _calc_heat_in_from_inlet_nodes(node, nodes_flag_dict)

            try:
                p_id = self.node_topology[node_id]['conns'][1][0]  # TODO: Заглушка что один родитель!
            except:
                p_id = self.node_topology[node_id]['conns'][0][0]

            p = self.node_topology[p_id]['obj']
            pipe_object = self.pipe_topology[(p_id, node_id)]
            # pipe_object = self.pipe_topology[(p_id, node_id)]
            sur_temp = self.surroundingTemperatureDegC
            htc = self.heatTransferCoefficientWToM2ToDegK
            pipe_object.q = q
            pipe_object.mfo = mfo
            pipe_object.mfw = mfw
            t_out, q_heat_out = pipe_object.calculate_temperature_drop(q_heat_in, sur_temp, htc, t_in)

            node.q = q
            node.mfo = mfo
            node.mfw = mfw
            node.t_in = t_in
            node.t_out = t_in
            node.q_heat_in = q_heat_in
            node.q_heat_out = q_heat_in

            nodes_flag_dict[node_id] = 1
            nodes_flag_dict = up_recursive_fun(p, nodes_flag_dict)
            return nodes_flag_dict

        ##############################################
        sources = self.wellnodes
        nodes_flag_dict = {i.id:0 for i in self.nodes}

        for s in sources:
            s_id = s.id
            if nodes_flag_dict[s_id]: continue

            # Для SOURCE входной тепловой поток задан температурой пласта
            try:
                p_id = self.node_topology[s_id]['conns'][1][0]  # TODO: Заглушка что один родитель!
            except:
                p_id = self.node_topology[s_id]['conns'][0][0]
            
            p = self.node_topology[p_id]['obj']
            pipe_object = self.pipe_topology[(p_id, s_id)] # TODO: Код падает если путь не в нужном порядке (поменять входной-выходной джойнт и все падает)
            # Обновляем свойства потока в трубопроводе

            sur_temp = self.surroundingTemperatureDegC
            htc = self.heatTransferCoefficientWToM2ToDegK
            q, mfo, mfw, t_in, q_heat_in = self._calc_source_heat_in(s)
            pipe_object.q = q
            pipe_object.mfo = mfo
            pipe_object.mfw = mfw
            t_out, q_heat_out = pipe_object.calculate_temperature_drop(q_heat_in, sur_temp, htc, t_in)

            s.t_in = t_in
            s.t_out = t_in
            s.q_heat_in = q_heat_in
            s.q_heat_out = q_heat_in

            nodes_flag_dict[s_id] = 1

            nodes_flag_dict = up_recursive_fun(p, nodes_flag_dict)


    def build_and_validate(self):
        """Build network topology, define initial values for every node and segment.
        Validate topology correctness.

        Returns 0 if everything's ok.
        """

        ## Initialize well rates
        ## Assumption: minimal liquid rate is 50
        ## Note: Too big rate result in big BHP, which can be bigger than reservoir pressure
        self.logger.info("Building tree topology")

        for node in self.nodes:
            q_gas = 0.0 * cu.CUB
            q_wat = 0.0 * cu.CUB
            q_oil = 0.0 * cu.CUB
            q_liq = 0.0 * cu.CUB
            flag = False

            if (
                node.ntype == NodeType.SOURCE and node.p_reservoir is not None
            ):  # when we use formula for calculation
                bhp_0 = node.p_reservoir - BHP_OFFSET * cu.ATM  # initial BHP guess, atm
                bhp_0 = bhp_0.to(u.bar)
                flag = True

                if node.ipr_curve is not None:
                    q_liq = node.ipr_curve.inv_func(bhp_0)

                if node.wct is not None:
                    q_wat = q_liq * node.wct
                    q_oil = q_liq - q_wat
                elif node.wct_curve is not None:
                    node.wct = node.wct_curve.func(bhp_0)
                    q_wat = q_liq * node.wct
                    q_oil = q_liq - q_wat
                    
                if node.wat_curve is not None:
                    q_wat = node.wat_curve.func(bhp_0)
                    

                if node.oil_curve is not None:
                    q_oil = node.oil_curve.func(bhp_0)

                if (node.gor is None or np.isnan(node.gor.value)) and node.gor_curve is not None:
                    print('wwwwwwwww')
                    node.gor = node.gor_curve(bhp_0)

                if node.gor is not None or np.isnan(node.gor.value):
                    q_gas = q_oil * node.gor
                else:
                    q_gas = node.get_qgas(bhp_0, wct=node.wct, qliq=q_liq)

                if hasattr(node, "gor_func"):
                    gor = node.gor_func(bhp_0, **node.gor_params)
                    q_gas = q_oil * gor
                    
                if node.gas_curve is not None:
                    q_gas = node.gas_curve.func(bhp_0)

                if (node.gor is None or np.isnan(node.gor.value)):
                    print('qqqqqqqqqqqqq')
                    node.gor = q_gas / q_oil

                if (node.wct is None or np.isnan(node.wct.value)):
                    node.wct = q_wat / q_liq
            else:
                ## if IPR tables of BHP and rates are given
                q_gas = START_Q * cu.CUB  # m^3/d
                q_oil = START_Q * cu.CUB  # m^3/d
                q_wat = START_Q * cu.CUB  # m^3/d
                flag = True

            # Mass rates, kg/s
            q_oil_m = self.rho_oil * q_oil.to(cu.M3_SEC)
            q_gas_m = self.rho_gas * q_gas.to(cu.M3_SEC)
            q_wat_m = self.rho_wat * q_wat.to(cu.M3_SEC)

            if flag and node.q == 0:
                node.q = q_gas_m + q_wat_m + q_oil_m  # Total mass rate, kg/s
                if node.q > MIN_TOTAL_RATE * cu.KG_SEC:
                    node.mfo = q_oil_m / node.q
                    node.mfw = q_wat_m / node.q

        for node in self.nodes:
            if node.ntype == NodeType.SOURCE and node.q < 0.0 * cu.KG_SEC:
                self.m_flagmomw = True
                self.m_flagmix = False

        for node in self.nodes:
            if self.m_flagtask:
                self.m_flagmomw = False
                self.m_flagmix = False
                node.mfw = self.m_mfw
                node.mfo = self.m_mfo

        p_max = 0.0 * u.Pa
        q_sum_wat = 0.0 * cu.KG_SEC
        q_sum_oil = 0.0 * cu.KG_SEC
        q_sum = 0.0 * cu.KG_SEC
        q_wats = np.zeros(len(self.nodes))
        q_oils = np.zeros(len(self.nodes))
        qs = np.zeros(len(self.nodes))
        for i, node in enumerate(self.nodes):
            ## find maximum pressure over all nodes
            if p_max < node.p:
                p_max = node.p
            elif node.q is None or np.isnan(node.q.value) or np.isclose(node.q.value, 0):  # if rate is not defined, continue
               continue
            
            qs[i] = node.q.value
            q_wats[i] = (node.q * node.mfw).value
            q_oils[i] = (node.q * node.mfo).value

        q_sum = math.fsum(qs) * cu.KG_SEC
        q_sum_wat = math.fsum(q_wats) * cu.KG_SEC
        q_sum_oil = math.fsum(q_oils) * cu.KG_SEC

        if q_sum.value == 0:  # rate is zero
            raise NetworkParameterNotSetError("Debit is not set!")
        if p_max.value == 0:  # there are nodes with given pressure
            raise NetworkParameterNotSetError("Pressure is not set!")

        self.m_mfw = q_sum_wat / q_sum  # average mass water fraction
        self.m_mfo = q_sum_oil / q_sum  # average mass oil fraction

        for node in self.nodes:
            if np.isnan(node.p.value) or np.isclose(node.p.value, 0):
                node.p = p_max
            if np.isnan(node.q.value) or np.isclose(node.q.value, 0):
                node.q = 0 * cu.KG_SEC
                node.mfw = self.m_mfw
                node.mfo = self.m_mfo
            node.nlink = 0  # nullify number of links for calculations below

        ## determine number and ids of pipes, connected to the node
        ## build topology of global network for nodes using pipe-end nodes ids
        for i, pipe in enumerate(self.pipes):
            n_0 = pipe.fnode[0]
            self.nodes[n_0].link[self.nodes[n_0].nlink] = i
            self.nodes[n_0].nlink += 1
            n_1 = pipe.fnode[1]
            self.nodes[n_1].link[self.nodes[n_1].nlink] = i
            self.nodes[n_1].nlink += 1

        if (
            not self.m_flagmomw
        ):  # for inverse flow the mixture is everywhere equal in composition
            for pipe in self.pipes:
                pipe.mfo = self.m_mfo
                pipe.mfw = self.m_mfw

        self.num_nodes = len(self.nodes)
        self.num_pipes = len(self.pipes)

        self.check_topology()

        ## pipelength is calculated by tube profile
        for pipe in self.pipes:
            pipe.l = 0 * u.m
            pipe.calculate_length()
            pipe.init_resistance(rho_wat=self.rho_wat)
            
        self.thp_dict_num = {i: pipe.fnode[1] for i, pipe in enumerate(self.pipes)}
        
        # self.calculate_mass_fraction()
        # self.calculate_pipe_rates()

        self.non_fixed_nodes = self.select_nodes(
            lambda n: n.ntype != NodeType.SINK, lambda n: n.id
        )
        
        print("Топология определена и проверена, значения начальных параметров для всех узлов заданы")

        # Для температурной модели и рекурсивного обхода
        self.node_topology, self.pipe_topology = self.define_new_topology_structure()

        return self

    def calculate_mass_fraction(self):
        """calculate mass fraction at every node."""
        if not self.m_flagmomw:
            return  # we don't mix in inverse flow because mass fractions will be equal everywhere

        iter_num = 1
        mm_sum = mw_sum = mo_sum = mm_node = 0.0
        err = 0.0

        while iter_num < NITER_MAS:
            err = 0  # divergence
            for node_id in range(self.num_nodes):
                node = self.nodes[node_id]

                ## if node is a source, pipe mass fractions are node mass fractions
                if node.ntype == NodeType.SOURCE:  # at node k rate is fixed
                    for i in range(node.nlink):
                        pipe = self.pipes[node.link[i]]  # neighbouring pipe
                        pipe.mfo = node.mfo
                        pipe.mfw = node.mfw
                    continue

                ## otherwise calculate node mass fraction
                ## calculate mixing proportional to input mass flow rates
                mm_sum = mw_sum = mo_sum = 0.0
                for i in range(node.nlink):
                    pipe = self.pipes[node.link[i]]  # neighbouring pipe
                    # id of neighbouring node
                    inode = get_neighbour_id(pipe.fnode, node_id)
                    nnode = self.nodes[inode]
                    mm_node = abs(pipe.q)  # mixture mass rate across pipe iflow
                    if ((node.p < nnode.p) or (nnode.ntype == NodeType.SOURCE)) and (
                        nnode.ntype != NodeType.SINK
                    ):  # flow goes from node k to node ilink
                        mm_sum += mm_node
                        mw_sum += pipe.mfw * mm_node
                        mo_sum += pipe.mfo * mm_node

                if mm_sum < MIN_MAS * cu.KG_SEC:
                    continue  # nothing goes thru node k

                # store previous node mass fractions
                mfw_prev = node.mfw
                mfo_prev = node.mfo

                # update node mass fractions
                node.mfw = mw_sum / mm_sum
                node.mfo = mo_sum / mm_sum

                ## for all output flows mass fractions are equal to mixture fractions in the node
                for i in range(node.nlink):
                    pipe = self.pipes[node.link[i]]  # neghbouring pipe
                    inode = get_neighbour_id(pipe.fnode, node_id)
                    nnode = self.nodes[inode]  # neighbour node
                    if nnode.ntype == NodeType.SINK or (node.p >= nnode.p):
                        pipe.mfw = node.mfw
                        pipe.mfo = node.mfo

                ## determine divergence this iteration
                err += abs(mfw_prev - node.mfw) + abs(mfo_prev - node.mfo)

            if err < TOL_MAS:
                break

            iter_num += 1

    def calculate_pipe_rates(self) -> None:
        """Calculate pipe rates"""
        for k in range(self.num_pipes):
            pipe = self.pipes[k]
            n_0 = self.nodes[pipe.fnode[0]]
            n_1 = self.nodes[pipe.fnode[1]]
            if n_0.ntype == NodeType.SOURCE:  # rate in node 0 is given
                pipe.q = abs(n_0.q)
            elif n_1.ntype == NodeType.SOURCE:  # rate in node 1 is given
                pipe.q = abs(n_1.q)
            else:
                pipe.q = abs((n_0.p - n_1.p) / pipe.res)
               
    def determine_thp_nodes(self):
        """Determine above lying node for every pipe."""
        for k, pipe in enumerate(self.pipes):
            i0, i1 = pipe.fnode[0], pipe.fnode[1]
            self.thp_dict_num[k] = i1 if self.nodes[i0].p_a > self.nodes[i1].p_a else i0

    def calculate_well_rate(self, bh_node: Node, 
                            th_node: Node, 
                            pipe: Pipe
        ) -> None:
        """Calculate and update well rate parameters."""
        q_gas, q_liq = 0.0, 0.0

        gor = bh_node.gor
        wct = bh_node.wct
        alq = pipe.alq  # artificial lift quantity
        bhp = bh_node.p_a
        kwargs = {
            "rho_oil": self.rho_oil,
            "rho_wat": self.rho_wat,
            "rho_gas": self.rho_gas,
            "alq": alq,
        }

        if bh_node.ipr_curve is not None:
            vfp_ipr_solver = VFP_IPR_METHODS[self.vfp_ipr_method]
            q_liq, bhp = vfp_ipr_solver(bh_node, th_node, pipe, **kwargs)
            gas_kwargs = {"gor": gor, "wct": wct, "qliq": q_liq}
            q_gas = bh_node.get_qgas(bhp, **gas_kwargs)
            
            ## mass flow rates
            q_wat_mass = (self.rho_wat * wct * q_liq).to(cu.KG_SEC)  # kg/s
            q_oil_mass = (self.rho_oil * (1 - wct) * q_liq).to(cu.KG_SEC)  # kg/s
            q_gas_mass = (self.rho_gas * q_gas).to(cu.KG_SEC)  # kg/s
            q_mass = q_wat_mass + q_oil_mass + q_gas_mass  # kg/s

            ## Update borehole node params
            bh_node.p_a = bhp
            bh_node.q = q_mass
            bh_node.mfw = q_wat_mass / q_mass
            bh_node.mfo = q_oil_mass / q_mass

        ## Update well pipe params
        pipe.q = bh_node.q
        pipe.mfo = bh_node.mfo
        pipe.mfw = bh_node.mfw

    def calculate_well_resistance(self, pipe: Pipe, d_h: u.Quantity) -> u.Quantity:
        n_0 = self.nodes[pipe.fnode[0]]
        n_1 = self.nodes[pipe.fnode[1]]
        try:
            if n_0.ntype == NodeType.SOURCE:
                self.calculate_well_rate(n_0, n_1, pipe)
                bhp, thp = n_0.p_a, n_1.p_a
            elif n_1.ntype == NodeType.SOURCE:
                self.calculate_well_rate(n_1, n_0, pipe)
                thp, bhp = n_0.p_a, n_1.p_a

            res = abs((bhp - thp) / pipe.q).to(cu.PA_KG_S)
            print(f"resistance {res}")
            return res if d_h < 0.0 else abs(res)
        except Exception as e:
            raise RuntimeError(
                f"Error while calculating tube resistance, pipe id: {pipe.id}"
            ) from e

    def calculate_tube_resistance(self, pipe: Pipe) -> u.Quantity:
        """Calculate resistance of a simple tube."""
        try:
            gor = 0 * cu.CUB_CUB
            if pipe.mfo > MIN_OIL_MASS_RATIO:
                gor = pipe.mfg * self.rho_oil / (pipe.mfo * self.rho_gas)
            th_node = self.nodes[self.thp_dict_num[pipe.id]]
            pp.pprint(th_node.p_a)
            thp = th_node.p_a  # THP, atm
            thp_plus_dp = thp + pipe.choke.dp.to(cu.ATM) if pipe.choke is not None else thp

            qwat = (pipe.q * pipe.mfw / self.rho_wat).to(cu.CUB)
            qoil = (pipe.q * pipe.mfo / self.rho_oil).to(cu.CUB)
            wct = qwat / (qwat + qoil)
            if not np.isfinite(wct.value):
                wct = 0.0 * cu.CUB_CUB
            qliq = qwat + qoil
            qgas = gor * qoil

            d = pipe.d.to(u.m) # Внутренний диаметр трубы

            d_h = pipe.profile[1][1] - pipe.profile[0][1]
            d_x = pipe.profile[1][0] - pipe.profile[0][0]
            l = (d_x**2 + d_h**2)**0.5
            print(f'l: {l}')
            print(f'd_h: {d_h}')
            angle = 90 - math.acos(abs(d_h) / l) * 180 / math.pi # Угол отклонения от вертикали
            
            r = pipe.r # Диаметр трубы 

            v_oil = qoil.to(cu.M3_SEC) / (math.pi * d ** 2 / 4) if qoil !=0 else 0 * cu.M_SEC
            v_gas = qgas.to(cu.M3_SEC) / (math.pi * d ** 2 / 4) if qgas !=0 else 0 * cu.M_SEC
            v_wat = qwat.to(cu.M3_SEC) / (math.pi * d ** 2 / 4) if qwat !=0 else 0 * cu.M_SEC

            thp_plus_dp = (thp_plus_dp).to(u.bar) # Давление на выходнном узле трубы

            t = pipe.t_out.to(cu.DEG_KELV, u.temperature()) # Температура на выходе трубы

            eps = r / d # отношение шероховатости к диаметру

            print(f't: {t}')
            print(f't_in = {pipe.t_in}, t_out: {pipe.t_out}')

            self.update_pvt(p=thp_plus_dp.to(u.Pa).value, t=t.value) #NOTE: нужно ли обновлять значения gor и wct?
            
            model_args = {
                'v_oil': v_oil.value, # м/с
                'v_wat': v_wat.value, # м/с
                'v_gas': v_gas.value, # м/с
                'rhoo': self.rho_oil.value, # кг/м3
                'rhow': self.rho_wat.value, # кг/м3
                'rhog': self.rho_gas.value, # кг/м3
                'wct': wct.value, # доли
                'muo': self.muo, # сПз (мПа / с)
                'muw': self.muw, # сПз (мПа / с)
                'mug': self.mug, # сПз (мПа / с)
                'gl_ift': 1, #TODO get from PVT module
                'diameter': d.value, # м
                'eps': eps.value, # безразмерное
                'angle': angle, # градусы
                'pressure': thp_plus_dp.value, # бар
                'p1_elevation': 1, #TODO get from ???
                'p2_friction': 1, #TODO get from ???
            }
            pp.pprint(model_args, sort_dicts=False)

            total_gradient = self.model(**model_args)['Total gradient']
            bhp = thp_plus_dp + total_gradient * l

            print(f'total_gradient: {total_gradient}')
            print(f'bhp: {bhp}')
            print(f'thp: {thp_plus_dp}')
            print('-------------------------------------------------------------------------')
            res = abs((bhp.to(u.Pa) - thp_plus_dp.to(u.Pa)) / pipe.q)

            return res if d_h < 0.0 else abs(res)

        except Exception as e:
            error_message = traceback.format_exc().splitlines() #Разбиваем на строки
            error_message = '\n'.join(error_message[1:]) # Убираем первую строку (повторяется в сообщении)
            raise PipeResistanceCalculationError(
                f"Error while calculating tube resistance, pipe number: {pipe.id}\n"
                f"Traceback:\n{error_message}"
            ) from e

    def set_node_pressures(self, x: np.ndarray, sym: bool) -> None:
        """Set node pressures according to linear system solution `x` """
        if sym:
            for j, k in enumerate(self.non_fixed_nodes):
                p = x[j] * u.Pa
                node = self.nodes[k]
                node.p = p
            return
        for k in range(self.num_nodes):
            p = x[k] * u.Pa
            node = self.nodes[k]
            node.p = p

    def update_node_pressures(self, x: np.ndarray, damping: float, sym: bool) -> None:
        """Update node pressures according to linear system solution `x`."""
        if sym:
            for j, k in enumerate(self.non_fixed_nodes):
                p = x[j] * u.Pa
                node = self.nodes[k]

                # node.p += damping * (p - node.p)
                node.p = (1 - damping) * node.p + damping * p
            return

        for k in range(self.num_nodes):
            p = x[k] * u.Pa
            node = self.nodes[k]
            node.p += damping * (p - node.p)
    
    def update_pvt(self, p: float | Iterable, t: float | Iterable):
        '''Функция пересчитывает pvt свойства и обновляет их
        :param p: давление, Па
        :param t: температура, К
        '''

        [self.bw, self.rho_wat, self.muw, self.hc_wat, self.salinity, self.z, self.bg, self.rho_gas,
        self.mug, self.hc_gas, self.pb, self.rs, self.compro, self.bo, self.rho_oil, self.muo, self.hc_oil,
        self.st_wat_gas, self.st_oil_gas, self.comprw, self.comprg] = self.pvt_model.calc_pvt(p, t).values()

        # Проверка и присваивание едининц измерения
        if hasattr(self.rho_wat, 'unit'):
            if self.rho_wat.unit == u.dimensionless_unscaled:
                self.rho_wat = self.rho_wat * cu.KG_M3
        else:
            self.rho_wat = self.rho_wat * cu.KG_M3

        if hasattr(self.rho_oil, 'unit'):
            if self.rho_oil.unit == u.dimensionless_unscaled:
                self.rho_oil = self.rho_oil * cu.KG_M3
        else:
            self.rho_oil = self.rho_oil * cu.KG_M3

        if hasattr(self.rho_gas, 'unit'):
            if self.rho_gas.unit == u.dimensionless_unscaled:
                self.rho_gas = self.rho_gas * cu.KG_M3
        else:
            self.rho_gas = self.rho_gas * cu.KG_M3

        print(f'rhoo: {self.rho_oil}\nrhow: {self.rho_wat}\nrhog: {self.rho_gas}')


    def fill_system(self, ybus: np.ndarray, rhs: np.ndarray, **kwargs) -> None:
        """Fill matrix and RHS vector of linear system."""
        ## formulate linear system of equations by defining JS = F
        # for j, k in enumerate(non_fixed_nodes):

        if kwargs.get("symmetrize", False):
            for j, k in enumerate(self.non_fixed_nodes):
                node = self.nodes[k]
                # in node k pressure is defined m_pipenodes[k].p
                if node.ntype == NodeType.SINK:
                    continue

                diag_arr = np.empty(node.nlink)

                for i in range(node.nlink):  # all connections to node k
                    ipipe = self.pipes[node.link[i]]  # number of flow going to k
                    ilink = (
                        ipipe.fnode[1] if ipipe.fnode[0] == k else ipipe.fnode[0]
                    )  # id of neighbour node
                    nnode = self.nodes[ilink]
                    y = 1.0 / ipipe.res  # tube hydrodynamic conductance, (kg/s)/atm
                    ## fill matrix of conductances

                    diag_arr[i] = y.value

                    try:
                        non_fixed_index = self.non_fixed_nodes.index(ilink)
                        ybus[j, non_fixed_index] -= y.value
                    except ValueError: # Значит это SINK
                        rhs[j] += (y * nnode.p).value
                
                ybus[j, j] = math.fsum(diag_arr)

                if node.ntype == NodeType.SOURCE:  # rate is fixed at node k
                    rhs[j] += node.q.value
            if kwargs.get("method", "general") == "qp":
                checkSPD(ybus)
            return
        for k in range(self.num_nodes):
            node = self.nodes[k]
            # in node k pressure is defined m_pipenodes[k].p
            if node.ntype == NodeType.SINK:  
                ybus[k, k] = 1.0
                rhs[k] = node.p.value
                continue

            for i in range(node.nlink):  # all connections to node k
                ipipe = self.pipes[node.link[i]]  # number of flow going to k
                ilink = (
                    ipipe.fnode[1] if ipipe.fnode[0] == k else ipipe.fnode[0]
                )  # id of neighbour node
                y = 1.0 / ipipe.res  # tube hydrodynamic conductance, (kg/s)/atm
                ## fill matrix of conductances
                ybus[k, k] += y.value
                ybus[k, ilink] -= y.value

            if node.ntype == NodeType.SOURCE:  # rate is fixed at node k
                rhs[k] += node.q.value

    def update_optimizables(self, x) -> None:
        """Update network parameters that are subject to optimization"""
        for j, choke in enumerate(self.autochokes):
            choke.dp = x[j] * cu.ATM

    def get_dim(self, symmetrize: bool = False) -> int:
        """Linear system of equations size."""
        return len(self.non_fixed_nodes) if symmetrize else self.num_nodes
    
    def update_tube_resistances(self, damping: float, iter_num: int) -> Tuple[float, float]:
        # -------------- CODE FOR TUBES -------------------------------------------    
        well_ids = []
        res_delta = np.empty(len(self.pipes))
        res_summa = np.empty(len(self.pipes))

        for i, pipe in enumerate(self.pipes):
            
            print(f'pipe_id: {pipe.id}\npipe_type: {pipe.ptype.name}\npipe_name: {pipe.name}\n')

            d_h = 0.0 * u.m
            prev_res = pipe.res  # prev resistance

            if pipe.ptype == PipeType.TUBE:
                curr_res = self.calculate_tube_resistance(pipe)
            elif pipe.ptype == PipeType.CONNECTOR:
                curr_res = RES_CONNECT * cu.PA_KG_S
            else:
                well_ids.append(i)
                continue

            ## calculate tolerance
            res_delta[i] = abs(curr_res - prev_res).value
            res_summa[i] = abs(curr_res).value

            ## Mix current and previous resistance with coefficient EKITER
            pipe.res = prev_res + damping * (curr_res - prev_res)

        # ------------ CODE FOR WELL RESISTANCE UPDATE ---------------------------
        if self.is_parallel:
            densities = {"rho_oil": self.rho_oil, 
                         "rho_wat": self.rho_wat, 
                         "rho_gas": self.rho_gas}
            # ------------PARALLEL----------------------------------------------------
            for idx in well_ids:
                well = self.pipes[idx]
                if iter_num == 1:
                    # assign connection pipe's parts to particular well
                    ps, ws = mp.Pipe(duplex=True)
                    self.senders[well.name] = ps
                    # run process for the well
                    p = mp.Process(target=create_well_process, args=(self.nodes, well, ws, damping, densities))
                    p.start()
                n_0 = self.nodes[well.fnode[0]]
                n_1 = self.nodes[well.fnode[1]]
                self.senders[well.name].send(["run", n_0.p, n_1.p])
            
            for idx in well_ids:
                well = self.pipes[idx]
                msg = self.senders[well.name].recv()
                prev_res = well.res
                curr_res = msg[0]
                res_delta[idx] = abs(curr_res - prev_res).value
                res_summa[idx] = abs(curr_res).value
                update_well_state(self.nodes, well, **msg[1])           
        else:        
            # --------------SEQUENTIAL-----------------------------------------------
            for idx in well_ids:
                well = self.pipes[idx]
                prev_res = well.res
                curr_res = self.calculate_well_resistance(well, d_h)
                res_delta[idx] = abs(curr_res - prev_res).value
                res_summa[idx] = abs(curr_res).value
                well.res = prev_res + damping * (curr_res - prev_res)    

        deltares = math.fsum(res_delta) * cu.PA_KG_S
        summares = math.fsum(res_summa) * cu.PA_KG_S 

        print(f'deltares: {deltares}')
        print(f'sumres: {summares}')


        return summares, deltares

    def generate_constraints(self, lb: Iterable[float], ub: Iterable[float]) -> None:
        for j, k in enumerate(self.non_fixed_nodes):
            node = self.nodes[k]
            if node.ntype != NodeType.SOURCE:
                continue

            q_lo, q_hi = node.q_b  # Node liquid rate bounds, m^3/d
            bhp_l = node.ipr_curve(q_hi).to(u.bar).value
            bhp_u = node.ipr_curve(q_lo).to(u.bar).value

            lb[j] = (np.maximum(bhp_l, 0) * u.bar).to(u.Pa).value
            ub[j] = (np.maximum(bhp_u, 0) * u.bar).to(u.Pa).value

    def update_well_chokes(
        self, lhs, rhs, dps: Iterable[float], damping: float = 0.5
    ) -> Tuple[float, float]:
        _ = lhs, rhs
        res_sum = np.empty(len(self.pipes))
        res_del = np.empty(len(self.pipes))
        for k, pipe in enumerate(self.pipes):
            if not pipe.choke:
                continue
            if not pipe.choke.is_auto:
                continue

            bh_node_id = (
                pipe.fnode[0]
                if pipe.fnode[1] == self.thp_dict_num[k]
                else pipe.fnode[1]
            )
            dp = pipe.choke.dp
            dp_ = dps[bh_node_id] * cu.ATM
            dp1 = damping * abs(dp_)

            pipe.choke.dp += dp1
            res_sum[k] = abs(dp1 + dp).value
            res_del[k] = abs(dp1)
        
        sumres = math.fabs(res_sum) * cu.ATM
        res = math.fabs(res_del) * cu.ATM
        return res, sumres

    def update_well_respres(self, dct: Dict[str,float]) -> None:
        for node in self.nodes:
            if node.ntype == NodeType.SOURCE and node.name in dct: 
                value = dct[node.name]
                if not hasattr(value, "unit"):
                    node.p_reservoir = value * cu.ATM
                else:
                    node.p_reservoir = value

    def reset_autochokes(self) -> None:
        for pipe in self.pipes:
            if pipe.choke:
                pipe.choke.reset()

    def fill_wellqliqs(self, qliqs: Iterable[float]) -> None:
        for i, well in enumerate(self.wellnodes):
            liq_rate_cubes = qliqs[i] * cu.CUB
            oil_debit = (liq_rate_cubes * (1 - well.wct) * self.rho_oil).to(cu.TON_DAY)
            wat_debit = (liq_rate_cubes * well.wct * self.rho_wat).to(cu.TON_DAY)
            gas_debit = (liq_rate_cubes * (1 - well.wct) * well.gor * self.rho_gas).to(
                cu.TON_DAY
            )

        if not any(
            v is None
            for v in (
                oil_debit,
                wat_debit,
                gas_debit,
            )
        ):
            debit = oil_debit + wat_debit + gas_debit
            well.q = debit.to(cu.KG_SEC)  # ton/d -> kg/sec
            well.mfo = oil_debit / debit
            well.mfw = wat_debit / debit

    def get_pressure_array(self):
        arr = np.zeros(len(self.non_fixed_nodes))
        
        for i, j in enumerate(self.non_fixed_nodes):
            arr[i] = self.nodes[j].p.value
        return arr