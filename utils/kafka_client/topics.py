from enum import Enum, auto


class _Fields(Enum):

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):  # это специальный метод
        return f"{name}_queue"


class Topics(_Fields):
    status = auto()
    tasks = auto()
    well_predictions = auto()
    well_tasks = auto()
