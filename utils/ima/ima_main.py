import datetime
import json
import os
import sys

from src.controllers.data_models.calc_params import CalcParams

sys.setrecursionlimit(10000)
sys.path.append('..')
sys.path.append('../mipt_solver')

import warnings
warnings.simplefilter("ignore")

from src.rate_model import Simulator
from src.mipt_solver.nodal import NetSolver, Network, read_network_from_json, read_solver_config_from_json

from src.ima.df_utils import *
from src.ima.plot_utils import *
from src.ima.model import IMAForecast # model wrapping around IMA calculations

from src.mipt_solver.nodal.metawell import model

from src.controllers.settings import service_settings


sys.modules['model'] = model


def ima_setup(data_path, params, topology):
    """
    Установка Интегрированной модели
    """
    mipt_net_path = f"{data_path}/topologies/Topology_3.json" ### FIXME: захардкожен путь до модели

    net = Network()

    # Вот сюда надо сделать чтение эндпоинта
    read_network_from_json(net, mipt_net_path, topology)
    
    net.build_and_validate()

    solver = NetSolver(net)
    read_solver_config_from_json(solver, mipt_net_path)

    rate_model = Simulator(weights_path=f"{data_path}/models/inflow")
    model = IMAForecast(rate_model=rate_model,
                        network=net,
                        net_solver=solver,
                        cfg_path=f"{data_path}/config_ima.json",
                        )
    
    model.load_params(params, data_path)
    # model.load_cfg()
    return model


def ima_calculation(model, report_step_dates: List[datetime.date]):
    """
    Запуск Интегрированной модели на расчет

    """
    model.prepare_forecast(report_step_dates=report_step_dates)
    model.forecast(test=0, to_display=False, to_save=False)


def comparison(model):
    """
    Сравнение результата расчета Интегрированной модели с реальными данными

    """
    # complete_data_with_cols(model.full_data)
    complete_data_with_cols(model.history) # Предсказанные значения

    # full_data = model.full_data[model.full_data.well.isin(model.well_names)]
    # history = model.history[model.history.well.isin(model.well_names)]
    # start_date = model.start_date
    # n_months = model
    # reldata, comp__ = make_reldata_and_comp(full_data, history, start_date)

    def get_value(obj, data, **kwargs):

        frmt_dict = {
            "exp_name": obj.cfg[EXP_NAME],
            "exp_date": obj.cfg[EXP_DATE],
            "n_years": obj.n_years,
            "n_months": obj.n_months,
            "type": PRED
        }

        frmt_dict.update(**kwargs)
        json_string = data.to_json(orient="records", date_format="iso")
        return json.loads(json_string)

    return get_value(model, model.history, type='full_pred')


def ima_meta(topology_path, params: CalcParams, topology):
    print('Running Solver...')
    model = ima_setup(topology_path, params=params, topology=topology)
    ima_calculation(model, params.report_step_dates)
    return comparison(model) ### FIXME: Запуск расчета не всегда подразумевает срванение с фактом 


def run_forecast(params, topology):
    topology_path = service_settings.data_path
    return ima_meta(topology_path, params=params, topology=topology)
