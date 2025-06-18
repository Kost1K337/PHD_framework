import json
from loguru import logger
from typing import Literal, Callable, List, NamedTuple, Optional

from kafka import KafkaProducer, KafkaConsumer, TopicPartition, OffsetAndMetadata
from kafka.consumer.fetcher import ConsumerRecord

from src.kafka_client.kafka_exceptions import MessageNotSendError


class TaskMsg(NamedTuple):
    correlation_id: str
    calc_params: dict
    topology: dict


class ParamData(NamedTuple):
    min_: float
    max_: float
    step: float


class WellTaskMsg(NamedTuple):
    correlation_id: str
    model_type: str
    Q: ParamData
    W: float
    G: float
    F: float
    Pt: float


def decode_well_task_msg(well_task_msg: ConsumerRecord) -> WellTaskMsg:
    try:
        decoded = json.loads(well_task_msg.value)
    except json.JSONDecodeError:
        raise

    def form_param_data(param_data: dict) -> Optional[ParamData]:
        if param_data is not None:
            return ParamData(param_data.get("min", 0),
                             param_data.get("max", 0),
                             param_data.get("step", 0))
        return None

    return WellTaskMsg(decoded.get("correlation_id"),
                       decoded.get("model_type", ""),
                       form_param_data(decoded.get("Q")),
                       decoded.get("W"),
                       decoded.get("G"),
                       decoded.get("F"),
                       decoded.get("Pt"))


def decode_task_msg(task_msg: ConsumerRecord) -> TaskMsg:
    """
    Декодирование сообщение из Kafka
    @param task_msg: Сообщение Kafka
    @return: Сообщение задания
    """
    try:
        decoded = json.loads(task_msg.value)
    except json.JSONDecodeError as e:
        raise e

    # в сообщении на отмену только id расчёта
    return TaskMsg(
        decoded.get("correlation_id", -1), decoded.get("calc_params", {}), decoded.get("topology", {})
    )


class KafkaClient:

    def __init__(
        self,
        bootstrap_servers: str,
        topics_get_task: Optional[List[str]] = None,
        topics_get_cancel: Optional[List[str]] = None,
        topics_get_well_task: Optional[List[str]] = None,
        topic_send_status: Optional[str] = None,
        topic_send_log: Optional[str] = None,
        topic_send_well_coordinates: Optional[str] = None,
        enable_auto_commit: bool = False,
        main_task_max_poll_interval_ms: int = 300000,
        main_task_max_poll_records: int = 500,
    ):
        """
        Инициализация клиентов для работы с Kafka.
        Создаётся Consumer для получения заданий и/или Producer для отправки сообщений с результатами.

        @param bootstrap_servers: Адрес для подключения к брокеру Kafka.
        @param topics_get_task: Топики для получения заданий.
        @param topics_get_cancel: Топики для отмен заданий.
        @param topic_send_log: Топик для отправки логов.
        @param enable_auto_commit: Коммитить offsets автоматически или в ручную. По умолчанию переделан в ручной режим.
        @raise NoBrokersAvailable: В случае отсутствия подключения к брокеру kafka.

        """

        # После начала эксплуатации будет понятно нужно ли менять auto_offset_reset
        # После начала эксплуатации будет понятно нужно ли использовать group_id
        # После начала эксплуатации будет понятно нужно ли менять enable_auto_commit

        self._consumer_task_topics = topics_get_task
        self._consumer_cancel_topics = topics_get_cancel
        self._consumer_well_task_topics = topics_get_well_task
        self._bootstrap_servers = bootstrap_servers

        # Авто коммит нужен для высоко нагруженных асинхронных систем.
        # Для долгих синхронных операций автокоммит не успевает срабатывать и происходит таймаут,
        # что приводит к перебалансировке и появлению дублей
        self._enable_auto_commit = enable_auto_commit

        # Подписка возможна только при наличии топиков для подписки
        if topics_get_task or topics_get_cancel:
            self._task_consumer = KafkaConsumer(
                bootstrap_servers=bootstrap_servers,
                group_id="metactive",
                max_poll_interval_ms=main_task_max_poll_interval_ms,
                max_poll_records=main_task_max_poll_records
            )
        else:
            self._task_consumer = None

        if topics_get_well_task:
            self._well_task_consumer = KafkaConsumer(
                bootstrap_servers=bootstrap_servers,
                group_id="well_metactive"
            )
        else:
            self._well_task_consumer = None

        self._send_status_topic = topic_send_status
        self._send_log_topic = topic_send_log
        self._send_coordinates_topic = topic_send_well_coordinates
        # Отправка результата возможна только при наличии топика для отправки
        if any((topic_send_status, topic_send_log, topic_send_well_coordinates)):
            # ключ не используется, если будет нужен то надо добавить сериализатор для строк.
            self._result_producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
        else:
            self._result_producer = None

    def send_well_coordinates(self,
                              correlation_id: str,
                              x: float,
                              y: float) -> None:
        msg_json = {
            "correlation_id": correlation_id,
            "x": x,
            "y": y
        }
        try:
            self._result_producer.send(self._send_coordinates_topic, msg_json)
            # logger.info(f"Сообщение {msg_json} отправлено в kafka")
        except MessageNotSendError:
            raise

    def send_well_coordinates_list(self,
                                   correlation_id: str,
                                   items: list) -> None:
        msg_json = {
            "correlation_id": correlation_id,
            "items": items
        }
        try:
            self._result_producer.send(self._send_coordinates_topic, msg_json)
        except MessageNotSendError:
            raise

    def send_status(self,
                    correlation_id: str,
                    status: Literal["done", "error", "started"],
                    message: Optional[str] = None) -> None:
        """
        Отправить сообщение со статусом
        Parameters
        ----------
        correlation_id: Уникальный ID расчета
        status: Статус
        message: Сообщение об ошибке/название json файла

        Returns
        -------

        """
        msg_json = {
            "correlation_id": correlation_id,
            "status": status,
            "message": message
        }
        try:
            self._result_producer.send(self._send_status_topic,
                                       msg_json)
            logger.info(f"Сообщение {msg_json} отправлено в kafka")
        except MessageNotSendError as e:
            raise e

    def _watch(
        self,
        topics: List[str],
        callback: Callable[[str, Optional[dict], Optional[dict]], None],
    ):
        """Подписка на получение новых сообщений в очереди kafka.

        Args:
            topics (List[str]): Названия топиков на которые надо подписаться.
            callback (Callable[[str, int], None]): Колбэк функция. Вызывается после получения задания.
        """

        if self._task_consumer:
            self._task_consumer.subscribe(topics)
        else:
            logger.error(
                "Не установлен consumer. Не возможно установить вотчер kafka"
            )
            return
        logger.info("Ждем сообщение из кафки")
        for msg in self._task_consumer:
            logger.info("Получено сообщение от kafka %s", msg.value)
            if not self._enable_auto_commit:
                self._commit(msg)
            if callback:
                decoded_task_msg = decode_task_msg(msg) if msg else None
                correlation_id = decoded_task_msg.correlation_id if decoded_task_msg else ""
                calc_params = decoded_task_msg.calc_params if decoded_task_msg else None
                topology = decoded_task_msg.topology if decoded_task_msg else None

                if (calc_params and topology) or correlation_id is not None:
                    stop_consumer = callback(
                        correlation_id, calc_params, topology
                    )
                    if stop_consumer:
                        break

    def _watch_well(self, topics: List[str], callback):
        if self._well_task_consumer:
            self._well_task_consumer.subscribe(topics)
        else:
            logger.error(
                "Не установлен consumer. Не возможно установить вотчер kafka"
            )
            return
        for msg in self._well_task_consumer:
            logger.info("Получено сообщение от kafka %s", msg.value)
            if not self._enable_auto_commit:
                self._commit(msg)
            if callback:
                decoded_well_task_msg = decode_well_task_msg(msg) if msg else None
                correlation_id = decoded_well_task_msg.correlation_id if decoded_well_task_msg else ""
                model_type = decoded_well_task_msg.model_type if decoded_well_task_msg else ""

                Q = decoded_well_task_msg.Q if decoded_well_task_msg else None
                W = decoded_well_task_msg.W if decoded_well_task_msg else None
                G = decoded_well_task_msg.G if decoded_well_task_msg else None
                F = decoded_well_task_msg.F if decoded_well_task_msg else None
                Pt = decoded_well_task_msg.Pt if decoded_well_task_msg else None

                if correlation_id:
                    callback(correlation_id, model_type, Q, W, G, F, Pt)

    def watch_tasks(self, callback: Callable[[str, Optional[dict], Optional[dict]], None]):
        """
        Подписка на получение новых заданий в очереди kafka.

        @param callback: Колбэк функция. Вызывается после получения задания.

        """
        self._watch(self._consumer_task_topics, callback)

    def watch_well_tasks(self, callback: Callable[[str, str, Optional[ParamData], Optional[float], Optional[float],
                                                   Optional[float], Optional[float]], None]):
        self._watch_well(self._consumer_well_task_topics, callback)

    def _commit(self, msg: ConsumerRecord):
        """
        Сохранение в kafka идентификаторов обработанных записей (offset).
        @param msg: Полученное сообщение от kafka.

        """

        # Разные группы могут читать сообщения из разных партиций в разной последовательности.
        # В рамках одной группы каждое сообщение будет прочитано только ОДНИМ consumer'ом,
        # остальные это сообщения не увидят.

        # Указываем для какой партиции и какого топика запомнить offset.
        # Привязку группы к партиции kafka делает автоматом для каждого топика.
        topic_partition = TopicPartition(msg.topic, msg.partition)
        if self._task_consumer:
            meta = self._task_consumer.partitions_for_topic(msg.topic)
            # Указывается с какого номера получать следующие сообщения. Номер текущей записи + 1
            options = {topic_partition: OffsetAndMetadata(msg.offset + 1, meta)}

            logger.info("Manual commit kafka %s", options)

            self._task_consumer.commit(options)
        else:
            logger.warning("Manual commit kafka failed!")

