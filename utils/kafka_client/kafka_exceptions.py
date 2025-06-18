class MessageNotSendError(Exception):
    def __init__(self, msg="Не удалось отправить сообщение в Kafka"):
        super().__init__(msg)
