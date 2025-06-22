from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import PolynomialFeatures

class PolyLinearRegression(LinearRegression):
    def __init__(self, degree=1, fit_intercept=True, copy_X=True, n_jobs=None, positive=False):
        super().__init__(fit_intercept=fit_intercept, copy_X=copy_X, n_jobs=n_jobs, positive=positive)
        self.degree=degree

    def fit(self, X, y):
        self.polynomial_transformer = PolynomialFeatures(degree=self.degree)
        X_poly = self.polynomial_transformer.fit_transform(X)
        super().fit(X_poly, y)
        return self

    def predict(self, X):
        X_poly = self.polynomial_transformer.transform(X)
        return super().predict(X_poly)
    

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from .linear_model import PolyLinearRegression


# Функционал качества модели
def my_custom_scorer(y_true, y_pred):
    diff = np.abs(y_true - y_pred)
    mask = list(map(lambda d, t: (d > 2) if t < 25 else ((d > 0.05*t) if t > 50 else (d > 0.1*t)), diff, y_true))
    res = (len(y_true)-np.count_nonzero(mask))/len(y_true)
    return res

# Параметры моделей
params = {
    'GradientBoostingRegressor': {'learning_rate': [0.05, 0.1], 'max_depth': [3, 4, 5, 6, 10, 15, 20],
                                  'min_samples_split': [2, 3, 4, 5], 'n_estimators': list(range(10, 105, 10)),
                                  'loss': ['absolute_error', 'huber'], 'criterion': ['friedman_mse', 'squared_error']},

    'CatBoostRegressor': {'depth': [2, 4, 6, 10, 15], 'learning_rate': [0.05, 0.1], 'iterations': [10, 30, 50, 100],
                          'verbose': [False], 'loss_function': ['MAE', 'RMSE']},

    'RandomForestRegressor': {'n_estimators': list(range(10, 105, 10)), 'max_depth': [3, 4, 5, 6, 10, 15, 20],
                              'min_samples_split': [2, 3, 4, 5], 'criterion': ['friedman_mse', 'squared_error']},

    'PolyLinearRegression': {'degree': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
}


class well_metamodel:
    '''
    Класс для создания и работы с моделью скважин

    Параметры
    ----------
    model_name : str
        Имя скважины, для которой строится модель
    models : list = [CatBoostRegressor]
        Модели, используемые в ансамбле [CatBoostRegressor, GradientBoostingRegressor, RandomForestRegressor]
    max_models : int = 5
        Максимальное количество моделей (не распространяется на линейную регрессию)
    degrees : list = [3, 4, 5, 6, 7, 8]
        Список степеней для полиномиальных фичей линейной регрессии
    scoring : func = 5% отклонение
        Функционал качества модели
    cv : int = 5
        Количесвто групп для кроссвалидации
    random_state : int = 42
        Начальное значение для генератора случайных чисел

    Свойства
    ----------
    model_name : str
        Имя скважины
    models : list
        Список типов моделей, входящих в ансамбль
    scoring : func
        Функционал качества модели
    workers : list
        Список моделей, вошедших в ансамбль
    weights : list
        Список весов применяемых моделей
    seed : int
        Начальное значение для генератора случайных чисел
    test_scores : list
        Средние точности на тестовых выборках в процессе создания ансамбля
    train_scores : list
        Средние точности на обучающих выборках в процессе создания ансамбля
    val_scores : list
        Точности модели на валидационной выборке в процессе обучения
    num_features : int
        Количество фичей. При фонтанной добыче 4, при механизированной добыче 5 (добавляется частота насоса или другйо параметр отвечающий за велечину лифта)

    Методы
    ----------
    fit
        Обучение модели
    predict
        Получение предсказания от модели
    adapt
        Адаптация модели на исторические данные
    plot_predict
        Построение кроссплота (Фактическое значение / Вычисленное значение)
    plot_learning_curve
        Построение кривой обучения модели

    Данные
    ----------
    X : [[float, float, float, float, float],]
        'Q', 'W', 'G', 'F', 'Pt' - Дебит жидкости (м3/сут), Обводненность (%), Газовый фактор (безрамерная), Частота насоса (Гц), Устьевое давление(атм)
    y : [[float],]
        'Pb' - Забойное давление (атм)
    '''
    def __init__(self,
                 well_name,
                 models=[PolyLinearRegression],
                 max_models=10,
                 scoring=my_custom_scorer,
                 cv=5,
                 random_state=42):
        self.well_name = well_name
        self.models = models
        self.max_models = max_models
        self.scoring = scoring
        self.cv = cv
        self.seed = random_state
        self._model_names = [model.__name__ for model in models]

        self.extrapolator = LinearRegression(positive=True)
        self.features_train_bounds = None
        self.main_estimator = None
        self.bias = 0
        self._workers_for_study = []
        self._is_fitted = False
        self.num_features = None
        self.test_scores = []
        self.train_scores = []
        self.val_scores = []

    def fit(self, X, y):
        '''
        Обучение модели

        Параметры
        ----------
        X : Pandas DataFrame
            Таблица фичей
        y : Pandas Series
            Строка таргетов
        '''
        if not self._is_fitted:
            self._is_fitted = True
            self.num_features = np.shape(X)[1]
            self.features_train_bounds = {col: (X[col].min(), X[col].max()) for col in X.columns}


            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=self.seed)

            self.extrapolator.fit(X, y)
            workers = self._select_models(X_train, y_train)
            for worker in workers:
                self._workers_for_study.append((f'{worker.get_params()["degree"]}th degree', worker))
                self.fit(X_train, y_train)
                self.val_scores.append(self.scoring(y_val, self.predict(X_val)))
        else:
            kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.seed)
            train_scores = []
            test_scores = []
            best_score = 0
            best_estimator = None
            for train_index, test_index in kf.split(X):
                X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                self.main_estimator = StackingRegressor(estimators=self._workers_for_study,
                                                        final_estimator=LinearRegression())
                self.main_estimator.fit(X_train, y_train)
                train_scores.append(self.scoring(y_train, self.main_estimator.predict(X_train)))
                test_scores.append(self.scoring(y_test, self.main_estimator.predict(X_test)))
                if test_scores[-1] > best_score:
                    best_estimator = self.main_estimator
                    best_score = test_scores[-1]

            self.main_estimator = best_estimator
            self.train_scores.append(np.mean(train_scores))
            self.test_scores.append(np.mean(test_scores))

    def predict(self, x):
        '''
        Получение предсказания от модели

        Параметры
        ----------
        x : Pandas DataFrame, list
            Фичи, на которые надо сделать предсказание

        Возвращаемое значение
        ----------
        y_pred : numpy array
            Список предсказанных значений
        '''
        if not self._is_fitted or self.main_estimator is None:
            raise ValueError("Model is not fitted yet. Please call `fit` before `predict`.")
        if isinstance(x, list):
            x = pd.DataFrame(data=x, columns=['Q', 'W', 'G', 'F', 'Pt'])

        in_bounds = all((x[col] >= bounds[0]).all() and (x[col] <= bounds[1]).all()
                    for col, bounds in self.features_train_bounds.items())

        if in_bounds:
            y_pred = self.main_estimator.predict(x) + self.bias
        else:
            y_pred = self.extrapolator.predict(x)
        
        return y_pred

    def export_to_onnx(self, initial_type):
        if not self._is_fitted or self.main_estimator is None:
            raise ValueError("Model is not fitted yet. Please call `fit` before exporting to ONNX.")
        model_onnx = convert_sklearn(self.main_estimator, initial_types=initial_type)
        return model_onnx
    
    def save_onnx(self, path):
        initial_type = [('float_input', FloatTensorType([None, self.num_features]))]
        onnx_model = self.export_to_onnx(initial_type)
        with open(path, "wb") as f:
            f.write(onnx_model.SerializeToString())

    def plot_crossplot(self, X, y_true):
        '''
        Строит кроссплот в координатах предсказанное значение / фактическое значение

        Параметры
        ----------
        X : Pamdas DataFrame
            Данные по которомым будет выполнено предсказание
        y : Pandas Series
            Фактические данные, соответсвующие данным из X
        '''
        y_pred = self.predict(X)
        plt.figure(figsize=(8, 8))
        plt.grid(True)
        plt.scatter(y_true, y_pred, s=0.2)
        upper_line, lower_line = [], []
        for y in y_true:
            if y < 25:
                upper_line.append(y + 2)
                lower_line.append(y - 2)
            elif y > 50:
                upper_line.append(y * 1.05)
                lower_line.append(y * 0.95)
            else:
                upper_line.append(y * 1.1)
                lower_line.append(y * 0.9)
        plt.plot(y_true, upper_line, '--', color='red')
        plt.plot(y_true, lower_line, '--', color='red')
        plt.plot(y_true, y_true, ':', color='black')
        plt.title(f'Скважина {self.well_name}')
        plt.ylabel('Предсказанное забойное давление, атм')
        plt.xlabel('Фактическое забойное давление, атм')
        plt.show()

    def plot_learning_curve(self):
        '''
        Строит кривую обучения модели
        '''
        plt.figure(figsize=(12, 8))
        plt.grid(True)
        plt.title(f'Скважина {self.well_name}')
        plt.plot(self.test_scores, label='test score')
        plt.plot(self.train_scores, label='train score')
        plt.plot(self.val_scores, label='validate score')
        plt.xticks(range(len(self._workers_for_study)), range(1, len(self._workers_for_study) + 1))
        plt.xlabel('Эпохи')
        plt.ylabel('Точность модели')
        plt.legend()
        plt.show()

    def _select_models(self, X, y):
        '''
        Производится отбор моделей, из которых будет собираться ансамбль

        Параметры
        ----------
        X : Pamdas DataFrame
            Данные по которомым будет выполнено предсказание
        y : Pandas Series
            Фактические данные, соответсвующие данным из X

        Возвращаемое значение
        ----------
        best_workers : list
            Список моделей с наивысшей точностью
        '''
        scorer = make_scorer(self.scoring, greater_is_better=True)
        scores = []
        workers = []
        for i in range(len(self.models)):
            grs = GridSearchCV(estimator=self.models[i](), param_grid=params[self._model_names[i]], cv=self.cv,
                               scoring=scorer, n_jobs=-1)
            grs.fit(X, y)
            s = np.array(grs.cv_results_['mean_test_score'])
            pars = np.array(grs.cv_results_['params'])
            pars = list(pars[~np.isnan(s)])
            s = list(s[~np.isnan(s)])
            scores.extend(s)
            workers.extend([self.models[i](**p) for p in pars])
        score_threshold = 0.95 * max(scores) if len(scores) > 0 else 0
        workers = [workers[i] for i in range(len(workers)) if scores[i] >= score_threshold]
        scores = [scores[i] for i in range(len(scores)) if scores[i] >= score_threshold]
        if len(scores) > self.max_models:
            ind = np.argpartition(scores, -self.max_models)[-self.max_models:]
            workers = np.array(workers)[ind]
            scores = np.array(scores)[ind]
        return workers

    def adapt(self, X_hist, y_hist):
        '''
        Адаптация модели на исторические данные

        Параметры
        ----------
        X_hist : Pandas DataFrame
            Данные по которомым будет выполнена адаптация
        y_hist : Pandas Series
            Исторические значения забойного давления
        '''
        self.bias = np.median(y_hist - self.predict(X_hist))