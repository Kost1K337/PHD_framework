import service.constants as const
import correlations._gas_correlations as gas
import correlations._oil_correlations as oil
import correlations._water_correlations as wat

from typing import Optional
import numpy as np
import torch
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from collections.abc import Iterable

class pvt_model:
    def __init__(
            self,
            gamma_gas: float,
            gamma_oil: float,
            gamma_wat: float,
            gor: float = 0,
            wct: float = None,
            oil_correlations: Optional[dict] = None,
            gas_correlations: Optional[dict] = None,
            water_correlations: Optional[dict] = None,
            pvt_type: str = 'full_dynamic'):
        """

        Parameters
        ----------
        :param gamma_gas: относительная плотность газа,
                        (относительно воздуха с плотностью 1.2217 кг/м3 при с.у.)
        :param gamma_oil: относительная плотность нефти,
                        (относительно воды с плотностью 1000 кг/м3 при с.у.)
        :param gamma_wat: относительная плотность воды,
                        (относительно воды с плотностью 1000 кг/м3 при с.у.)
        :param gor: газовый фактор, м3/м3
        :param wct: обводненность, д.ед.
        :param oil_correlations: корреляции для нефти
        :param gas_correlations: корреляции для газа
        :param water_correlations: корреляции для воды
        :param pvt_type: тип расчета pvt
            'full_dymanic' - полный расчет pvt на каждом шаге
            'simple_dynamic' - расчет pvt на каждом шаге по предрасчитанным значениям
            'static' - фиксированные параметры pvt
        -------
        """
        if water_correlations is None:
            water_correlations = const.WAT_CORRS
        if gas_correlations is None:
            gas_correlations = const.GAS_CORRS
        if oil_correlations is None:
            oil_correlations = const.OIL_CORRS

        self.gamma_gas = gamma_gas
        self.gamma_oil = gamma_oil
        self.gamma_wat = gamma_wat
        self.wct = wct
        self.gor = gor
        self.wct_init = wct
        self.gor_init = gor
        self._oil_correlations = self.__check_correlations(
            {k: v.lower() for k, v in oil_correlations.items() if v is not None},
            const.OIL_CORRS,
        )
        self._gas_correlations = self.__check_correlations(
            {k: v.lower() for k, v in gas_correlations.items() if v is not None},
            const.GAS_CORRS,
        )
        self._water_correlations = self.__check_correlations(
            {k: v.lower() for k, v in water_correlations.items() if v is not None},
            const.WAT_CORRS,
        )

        # Инициализация атрибутов класса BlackOilModel
        self.pb = None
        self.rs = None
        self.muo = None
        self.mul = None
        self.rho_oil = None
        self.bo = None
        self.compro = None
        self.z = None
        self.bg = None
        self.rho_gas = None
        self.mug = None
        self.bw = None
        self.comprw = None
        self.rho_wat = None
        self.muw = None
        self.salinity = None
        self.st_wat_gas = None
        self.st_oil_gas = None
        self.st_liq_gas = None
        self.hc_wat = None
        self.hc_gas = None
        self.hc_oil = None
        self.mum = None
        self.comprg = None
        self.oil_corrs = oil.OilCorrelations(self._oil_correlations)
        self.gas_corrs = gas.GasCorrelations(self._gas_correlations)
        self.wat_corrs = wat.WaterCorrelations(self._water_correlations)

        self.__define_pvt_funcs()

        self.pvt_type = pvt_type

        if self.pvt_type == 'simple_dynamic':
            self.__initialize_interpolate()
        elif self.pvt_type == 'static':
            self.init_stat = False


    def __define_pvt_funcs(self):
        """
        Определение функций для расчета PVT-свойств в зависимости от количества поданных таблично
        """
        self.table_model = None

        for k in const.ALL_PROPERTIES:
            if not hasattr(self, k + "_func"):
                setattr(
                    self,
                    k + "_func",
                    getattr(
                        getattr(self, const.ALL_PROPERTIES[k][0]),
                        const.ALL_PROPERTIES[k][1],
                    ),
                )

    
    @staticmethod
    def __check_correlations(correlations: dict, correlations_default: dict) -> dict:
        """ "
        Функция проверки корреляций свойств

        :param correlations: словарь корреляций для проверки
        :param correlations_default: словарь корреляций по умолчанию

        :return: correlations - скорректированный словарь корреляций
        """
        for key in correlations_default:
            if correlations.get(key) is None:
                correlations.update({key: correlations_default[key]})
        return correlations

    @property
    def oil_correlations(self):
        """
        read-only атрибут с набором корреляций для расчета нефти

        :return: словарь с набором корреляций для расчета нефти
        -------

        """
        return self._oil_correlations

    @property
    def gas_correlations(self):
        """
        read-only атрибут с набором корреляций для расчета газа

        :return: словарь с набором корреляций для расчета газа
        -------

        """
        return self._gas_correlations

    @property
    def water_correlations(self):
        """
        read-only атрибут с набором корреляций для расчета воды

        :return: словарь с набором корреляций для расчета воды
        -------

        """
        return self._water_correlations
    

    def __initialize_interpolate(self, p_start: float = 1, p_stop: float = 400, t_start: float = 0, t_stop: float = 200):
        """
        Инициализация таблицы для интерполирования

        Parameters
        ----------
        :param p_start: минимальное давление, атм
        :param p_stop: максимальное давление, атм
        :param t_start: минимальная температура, C
        :param t_stop: максимальная температура, C
        """
        self.__max_p = p_stop
        self.__min_p = p_start
        self.__max_t = t_stop
        self.__min_t = t_start

        p_values = np.arange(p_start, p_stop+1, 5) * 101325
        t_values = np.arange(t_start, t_stop+1, 5) + 273

        pp, tt = np.meshgrid(p_values, t_values)

        z = []

        for p, t in zip(pp.flatten(), tt.flatten()):
            pvt_values = self.__calc_pvt_default(p, t, list(const.ALL_PROPERTIES.keys()))
            z.append(pvt_values)

        self.__interpolator = LinearNDInterpolator(np.vstack([pp.flatten(), tt.flatten()]).T, np.array(z))
        self.__exrtapolator = NearestNDInterpolator(np.vstack([pp.flatten(), tt.flatten()]).T, np.array(z), rescale=True)
        self.__interp_values_index = dict([(list(const.ALL_PROPERTIES.keys())[i], i) for i in range(len(const.ALL_PROPERTIES.keys()))])
    

    def calc_pvt(self, p: float | Iterable, t: float | Iterable, params: list = list(const.ALL_PROPERTIES.keys())):
        """
        Метод расчета PVT-параметров нефти, газа и воды

        Parameters
        ----------
        Функция для расчета всех физико-химических свойств
        :param p: давление, Па
        :param t: температура, К
        :param params: список параметров, который необходимо вернуть как реультат работы функции

        :return: pvt-параметры для нефти, газа и воды
        -------
        """
        if not isinstance(t, Iterable) or not isinstance(p, Iterable):
            if not (isinstance(t, Iterable) or isinstance(p, Iterable)):
                t = np.array([t])
                p = np.array([p])
            else:
                if not isinstance(t, Iterable):
                    t = (np if not isinstance(p, torch.Tensor) else torch).full_like(p, t)
                if not isinstance(p, Iterable):
                    p = (np if not isinstance(t, torch.Tensor) else torch).full_like(t, p)

        result = []
        for p_value, t_value in zip(p, t):
            if self.pvt_type == 'full_dynamic':
                result.append(self.__calc_pvt_default(p_value, t_value, params))
            elif self.pvt_type == 'simple_dynamic':
                if not isinstance(p, torch.Tensor):
                    result.append(self.__calc_pvt_interp(p_value, t_value, params))
                else:
                    result.append(self.__calc_pvt_interp(p_value.detach().numpy(), t_value.detach().numpy(), params))
            elif self.pvt_type == 'static':
                result.append(self.__calc_pvt_static(p, t, params))

        result = np.array(result) if not isinstance(p, torch.Tensor) else torch.Tensor(result)
        
        return [result[:, i] for i in range(len(params))]


    def __calc_pvt_default(self, p: float, t: float, params: list = None):
        """
        Метод расчета PVT-параметров нефти, газа и воды

        Parameters
        ----------
        Функция для расчета всех физико-химических свойств
        :param p: давление, Па
        :param t: температура, К
        :param params: список параметров, который необходимо вернуть как реультат работы функции

        :return: pvt-параметры для нефти, газа и воды
        -------
        """
        self.__calc_water_pvt_parameters(p, t)
        self.__calc_gas_pvt_parameters(p, t)
        self.__calc_oil_pvt_parameters(p, t)

        if not self.wct is None:
            self.__calc_liq_pvt_parameters(p, t)

        return [getattr(self, param) for param in params]
        

    def __calc_pvt_interp(self, p: float, t: float, params: list = None):
        """
        Метод для расчета свойств по таблице

        Parameters
        ----------
        Функция для расчета всех физико-химических свойств
        :param p: давление, Па
        :param t: температура, К
        :param params: список параметров, который необходимо вернуть как реультат работы функции

        :return: pvt-параметры для нефти, газа и воды  
        """
        result = []

        if (self.__min_p <= p <= self.__max_p) and (self.__min_t <= t <= self.__max_t):
            values = self.__interpolator(np.array([p, t])).flatten()
        else:
            values = self.__exrtapolator(np.array([p, t])).flatten()

        for param in params:
            result.append(values[self.__interp_values_index[param]])
        return result
    

    def __calc_pvt_static(self, p: float, t: float, params: list = None):
        """
        Метод для получения статических свойств

        Parameters
        ----------
        Функция для расчета всех физико-химических свойств
        :param p: давление, Па
        :param t: температура, К
        :param params: список параметров, который необходимо вернуть как реультат работы функции

        :return: pvt-параметры для нефти, газа и воды  
        """
        if not self.init_stat:
            self.init_stat = True
            self.__calc_pvt_default((np if not isinstance(p, torch.Tensor) else torch).mean(p), (np if not isinstance(p, torch.Tensor) else torch).mean(t), params)

        return [getattr(self, param) for param in params]


    def __calc_water_pvt_parameters(self, p: float, t: float):
        """
        Метод расчета PVT - параметров для воды

        Parameters
        ----------
        :param p: давление, Па
        :param t: температура, К
        :return: метод изменяет атрибуты класса BlackOil, отвечающие за pvt-свойства воды
        -------
        """
        self.bw = self.bw_func(p=p, t=t, pvt_property="bw")
        self.salinity = self.salinity_func(p=p, t=t, gamma_wat=self.gamma_wat, pvt_property="salinity")
        self.rho_wat = self.rho_wat_func(
            t=t,
            p=p,
            pvt_property="rho_wat",
            gamma_wat=self.gamma_wat,
            bw=self.bw,
            salinity=self.salinity,
        )
        self.muw = self.muw_func(p=p, t=t, pvt_property="muw", salinity=self.salinity)
        self.hc_wat = self.hc_wat_func(t=t, p=p, pvt_property="hc_wat")
        self.st_wat_gas = self.st_wat_gas_func(p=p, t=t, pvt_property="st_wat_gas")
        self.comprw = self.comprw_func(p=p, t=t, salinity=self.salinity, pvt_property="comprw")

    def __calc_gas_pvt_parameters(self, p: float, t: float):
        """
        Метод расчета PVT - параметров для газа

        Parameters
        ----------
        :param p: давление, Па
        :param t: температура, К
        :return:метод изменяет атрибуты класса BlackOil, отвечающие за pvt-свойства газа
        -------
        """
        self.z = self.z_func(p=p, t=t, gamma_gas=self.gamma_gas, pvt_property="z")
        self.bg = self.bg_func(p=p, t=t, z=self.z, pvt_property="bg")
        self.rho_gas = self.rho_gas_func(p=p, t=t, pvt_property="rho_gas", gamma_gas=self.gamma_gas, bg=self.bg)
        self.mug = self.mug_func(p=p, t=t, pvt_property="mug", gamma_gas=self.gamma_gas, rho_gas=self.rho_gas)
        self.hc_gas = self.hc_gas_func(p=p, t=t, gamma_gas=self.gamma_gas, pvt_property="hc_gas")
        self.comprg = self.comprg_func(p=p, t=t, z=self.z)

    def __calc_oil_pvt_parameters(self, p: float, t: float):
        """
        Метод расчета PVT - параметров для нефти

        Parameters
        ----------
        :param p: давление, Па
        :param t: температура, К
        :return: метод изменяет атрибуты класса BlackOil, отвечающие за pvt-свойства нефти
        -------
        """
        rp = self.gor_init

        if rp == 0:
            self.pb = const.ATM
            self.rs = 0
            rsb = 0
        else:
            self.pb = min(
                self.oil_corrs.calc_pb(t, rp, self.gamma_oil, self.gamma_gas),
                const.PB_MAX,
            )
            rp = self.oil_corrs.calc_rs(self.pb, t, self.gamma_oil, self.gamma_gas)

            self.gor = rp

            rsb = self.oil_corrs.calc_rs(self.pb, t, self.gamma_oil, self.gamma_gas)
            self.pb = min(self.pb, const.PB_MAX)

            if p < self.pb:
                self.rs = self.oil_corrs.calc_rs(p, t, self.gamma_oil, self.gamma_gas)
            else:
                self.rs = rsb

        self.compro = self.compro_func(
            t=t,
            p=p,
            pvt_property="compro",
            gamma_oil=self.gamma_oil,
            gamma_gas=self.gamma_gas,
            rsb=self.rs,
        )

        # Вызов метода расчета объемного коэффициента нефти
        if p <= self.pb:
            self.bo = self.oil_corrs.calc_oil_fvf(
                p, t, self.rs, self.gamma_oil, self.gamma_gas, self.compro, self.pb
            )
        else:
            self.bo = self.oil_corrs.calc_oil_fvf(
                p,
                t,
                self.rs,
                self.gamma_oil,
                self.gamma_gas,
                self.compro,
                self.pb,
                self.bob(t, self.pb, rsb, self.compro),
            )

        self.rho_oil = self.rho_oil_func(
            p=p,
            t=t,
            pvt_property="rho_oil",
            rs=self.rs,
            bo=self.bo,
            gamma_oil=self.gamma_oil,
            gamma_gas=self.gamma_gas,
        )

        self.muo = self.muo_func(
            p=p,
            t=t,
            pvt_property="muo",
            pb=self.pb,
            gamma_oil=self.gamma_oil,
            rs=self.rs,
            calibr_mu=1,
        )
        self.hc_oil = self.hc_oil_func(t=t, gamma_oil=self.gamma_oil, p=p, pvt_property="hc_oil")
        self.st_oil_gas = self.st_oil_gas_func(
            t=t, p=p, pvt_property="st_oil_gas", gamma_oil=self.gamma_oil, rs=self.rs
        )

    def __calc_liq_pvt_parameters(self, p: float, t: float):
        """
        Метод расчета PVT - параметров для жидкости

        Parameters
        ----------
        :param p: давление, Па
        :param t: температура, К
        :return: метод изменяет атрибуты класса BlackOil, отвечающие за pvt-свойства жидкости
        -------
        """

        self.st_liq_gas = self.__calc_stlg(self.st_oil_gas, self.st_wat_gas)
        self.mul = self.__calc_mul(self.muo, self.muw, self.wct)

    def bob(self, t: float, pb: float, rsb: float, compro: float) -> float:
        """
        Метод расчета объемного коэффициента нефти при давлении насыщения

        Parameters
        ----------
        :param t: температура, К
        :param pb: давление насыщения, Па
        :param rsb: газосодержание при давлении насыщения, м3/м3
        :param compro: сжимаемость нефти, 1/Па

        :return: объемный коэффициент нефти при давлении насыщения, м3/м3
        -------
        """
        return self.oil_corrs.calc_oil_fvf(pb, t, rsb, self.gamma_oil, self.gamma_gas, compro, pb)
    
    def __calc_stlg(self, sto, stw):
        """
        Метод расчета поверхностного натяжения на границе газ-жидкость

        Parameters
        ----------
        :param sto: поверхностное натяжение на границе газ-нефть, Н/м
        :param stw: поверхностное натяжение на границе газ-вода, Н/м

        :return: st (superficial tension) - поверхностное натяжение, Н/м
        -------
        """
        if sto is not None and stw is not None:
            wc_rc = self.wct
            # В пайпсим написано что есть переход в 60 %,
            # но там ошибка в мануале на самом деле всегда так
            st = sto * (1 - wc_rc) + stw * wc_rc
        else:
            st = None
        return st
    
    def __calc_mul(
        self,
        muo: float,
        muw: float,
        wct: float,
    ):
        """
        Метод расчета вязкости жидкости

        Parameters
        ----------
        :param muo: вязкость нефти, сПз
        :param muw: вязкость воды, сПз
        :param wct: обводненность, д.ед

        :return: вязкость жидкости, сПз
        -------
        """
        if muo is None:
            muo = 0
        if muw is None:
            muw = 0
        return muo * (1 - wct) + muw * wct
