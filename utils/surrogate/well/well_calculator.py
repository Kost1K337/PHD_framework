import pickle
from typing import List


class WellCalculator:
    def __init__(self, model_path: str) -> None:
        self._model = self._unpickle_model(model_path)

    @staticmethod
    def _unpickle_model(model_path: str) -> pickle.OBJ:
        """
        Распаковка объекта
        Parameters
        ----------
        model_path: Путь до pickle-файла модели

        Returns
        -------
        Объект модели
        """
        try:
            return pickle.load(open(model_path, "rb"))
        except pickle.UnpicklingError as e:
            raise e

    def predict(self, params: List[float]) -> float:
        """
        Предсказание
        Parameters
        ----------
        params Список параметров

        Returns
        -------
        Значение предсказания
        """
        try:
            return self._model.predict(params)[0]
        except ValueError as ex:
            if ex.args[0] == 'X has 5 features, but LinearRegression is expecting 4 features as input.':
                return self._model.predict(params[:4])[0]
        except IndexError:
            raise Exception("Невалидное значение в одном из переданных параметров")
        except Exception as e:
            raise e
