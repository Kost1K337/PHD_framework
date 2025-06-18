from enum import Enum, auto


class _Fields(Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()


class Statuses(_Fields):
    ERROR = auto()
    DONE = auto()
    STARTED = auto()
