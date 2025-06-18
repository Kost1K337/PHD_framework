import os.path
import threading
from typing import Optional

import uvicorn

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from src.controllers.data_models.calc_params import CalcParams
from src.ima.ima_main import run_forecast
from src.kafka_client.kafka_client import KafkaClient, ParamData
from src.kafka_client.statuses import Statuses
from src.kafka_client.topics import Topics
from src.controllers.settings import service_settings
from src.s3_client.s3_client import S3Client
from src.well_calculator.well_calculator import WellCalculator

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "healthy"}


kafka_client = KafkaClient(bootstrap_servers=service_settings.kafka_bootstrap_servers,
                           topics_get_task=[Topics.tasks.value],
                           topics_get_well_task=[Topics.well_tasks.value],
                           topic_send_status=Topics.status.value,
                           topic_send_well_coordinates=Topics.well_predictions.value,
                           main_task_max_poll_interval_ms=service_settings.main_task_max_poll_interval_ms,
                           main_task_max_poll_records=service_settings.main_task_max_poll_records)

s3_client = S3Client(storage_url=service_settings.storage_url,
                     result_bucket=service_settings.result_bucket,
                     access_key=service_settings.storage_auth_client_id,
                     secret_key=service_settings.storage_auth_secret_key)


def params_gen(min_: float, max_: float, step: float):
    i = min_
    while i < max_:
        yield i
        i += step
        if i >= max_:
            yield max_


def run_calculation(correlation_id: str, calc_params: dict, topology: dict) -> None:
    """
    Запуск расчета
    Parameters
    ----------
    correlation_id: Уникальный ID расчета
    calc_params: Параметры расчета
    topology: Топология

    Returns
    -------

    """
    try:
        kafka_client.send_status(correlation_id, Statuses.STARTED.value, None)
        calc_params = CalcParams(**calc_params)
        result = run_forecast(calc_params, topology)
        s3_client.push_to_result_bucket(result, correlation_id)
        kafka_client.send_status(correlation_id, Statuses.DONE.value, f"Расчет {correlation_id} завершен")
    except Exception as e:
        kafka_client.send_status(correlation_id, Statuses.ERROR.value, str(e))


def run_prediction(correlation_id: str,
                   model_type: str,
                   q_data: ParamData,
                   *args):
    """
    Запуск предсказания
    Parameters
    ----------
    correlation_id: Уникальный ID расчета
    model_type: Тип модели
    q_data: Дебит жидкости
    args: Фичи

    Returns
    -------

    """
    try:
        model_path = os.path.join(service_settings.predict_model_path, f"{model_type}.pickle")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модели {model_type} не существует")
        well_calculator = WellCalculator(model_path)
        kafka_client.send_status(correlation_id, Statuses.STARTED.value, None)
        predict_params = [v for v in args if v is not None]
        if not q_data:
            raise
        predicted_values = []
        for q in params_gen(q_data.min_, q_data.max_, q_data.step):
            predicted_values.append({"y": well_calculator.predict([q, *predict_params]), "x": q})
        kafka_client.send_well_coordinates_list(correlation_id, predicted_values)
        kafka_client.send_status(correlation_id, Statuses.DONE.value, "Plot data predicted")
    except Exception as e:
        kafka_client.send_status(correlation_id, Statuses.ERROR.value, str(e))


def run_kafka_calculation(correlation_id: str, calc_params: dict, topology: dict) -> None:
    """
    Запуск расчета в отдельном потоке
    Parameters
    ----------
    correlation_id: Уникальный ID расчета
    calc_params: Параметры расчета
    topology: Топология

    Returns
    -------

    """
    m_thread = threading.Thread(
        target=run_calculation, args=(correlation_id, calc_params, topology)
    )
    m_thread.name = f"calc-{correlation_id}"
    m_thread.start()
    m_thread.join()


def run_kafka_prediction(correlation_id: str,
                         model_type: str,
                         q: Optional[ParamData],
                         w: Optional[float],
                         g: Optional[float],
                         f: Optional[float],
                         pt: Optional[float]):
    """
    Запуск предсказания графика в потоке
    Parameters
    ----------
    correlation_id: Уникальный ID расчета
    model_type: Тип модели (номер pickle-файла)
    q: Дебит жидкости (изменяемая величина)
    w: Обводнение
    g: Газовый фактор
    f: Частота
    pt: Забойное давление

    Returns
    -------

    """
    m_thread = threading.Thread(
        target=run_prediction, args=(correlation_id, model_type, q, w, g, f, pt)
    )
    m_thread.name = f"predict-{correlation_id}"
    m_thread.start()
    m_thread.join()


@app.on_event("startup")
async def startup():
    t1 = threading.Thread(target=kafka_client.watch_tasks, args=(run_kafka_calculation,))
    t2 = threading.Thread(target=kafka_client.watch_well_tasks, args=(run_kafka_prediction,))
    t1.start()
    t2.start()


@app.on_event("shutdown")
async def shutdown():
    # cleanup
    pass


@app.get("/")
def redirect_docs() -> None:
    """Редирект на SwaggerUI."""
    return RedirectResponse(url="/docs")


class App:
    @staticmethod
    def run():
        uvicorn.run("app:app", port=8000, host="0.0.0.0", reload=False, workers=1)
