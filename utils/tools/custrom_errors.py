class TypeNotSupportedError(BaseException):
    def __init__(self, type_: str):
        super().__init__(f"Тип {type_} не поддерживаются")


class SolverError(BaseException):
    def __init__(self, msg: str):
        super().__init__(msg)
