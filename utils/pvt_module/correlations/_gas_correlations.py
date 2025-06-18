"""
Модуль, для описания корреляций для расчета свойств газа
"""
import numpy as np
import torch

import scipy.optimize as opt


class GasCorrelations:
    """
    Класс, включающий функции для расчета газовых свойств от давления и температуры
    """

    __slots__ = ["ppc", "tpc", "z", "mu", "hc_gas"]

    def __init__(self, gas_correlations: dict):
        """
        :param gas_correlations: словарь с набором корреляций для каждого свойства
        """
        if gas_correlations["ppc"] == "standing":
            self.ppc = self.__pseudocritical_pressure_standing
        else:
            raise ValueError(
                f"Корреляция {gas_correlations['ppc']} для ppc пока не реализована." f"Используйте другую корреляцию",
            )

        if gas_correlations["tpc"] == "standing":
            self.tpc = self.__pseudocritical_temperature_standing
        else:
            raise ValueError(
                f"Корреляция {gas_correlations['tpc']} для tpc пока не реализована." f"Используйте другую корреляцию",
            )

        if gas_correlations["z"] == "kareem":
            self.z = self.__z_kareem
        elif gas_correlations["z"] == "dranchuk":
            self.z = self.__z_dranchuk
        elif gas_correlations["z"] == "standing":
            self.z = self.__z_standing
        else:
            raise ValueError(
                f"Корреляция {gas_correlations['z']} для Z пока не реализована." f"Используйте другую корреляцию",
            )

        if gas_correlations["mu"] == "lee":
            self.mu = self.__gas_viscosity_lee
        else:
            raise ValueError(
                f"Корреляция {gas_correlations['mu']} для вязкости газа "
                f"пока не реализована. "
                f"Используйте другую корреляцию",
            )

        if gas_correlations["hc"] == "mahmood":
            self.hc_gas = self.__calc_heat_capacity_mahmood
        elif gas_correlations["hc"] == "const":
            self.hc_gas = self.__calc_heat_capacity_const
        else:
            raise ValueError(
                f"Корреляция {gas_correlations['hc']} для удельной теплоемкости газа "
                f"пока не реализована. "
                f"Используйте другую корреляцию",
            )

    @staticmethod
    def __pseudocritical_temperature_standing(gamma_gas: float) -> float:
        """
        Метод расчета псевдокритической температуры по корреляции Standing

        Parameters
        ----------
        :param gamma_gas: относительная плотность газа, доли,
        (относительно в-ха с плотностью 1.2217 кг/м3 при с.у.)

        :return: псевдокритическая температура, К
        -------
        """
        pc_t_standing = 93.33333333333333 + 180.55555555555554 * gamma_gas - 6.944444444444445 * (gamma_gas**2)
        return pc_t_standing

    @staticmethod
    def __pseudocritical_pressure_standing(gamma_gas: float) -> float:
        """
        Метод расчета псевдокритического давления по корреляции Standing

        Parameters
        ----------
        :param gamma_gas: относительная плотность газа, доли,
        (относительно в-ха с плотностью 1.2217 кг/м3 при с.у.)

        :return: псевдокритическое давление, Па
        -------
        """
        pc_p_standing = 4667750.68747498 + 103421.3593975254 * gamma_gas - 258553.39849381353 * (gamma_gas**2)
        return pc_p_standing

    @staticmethod
    def __dak_func(z, ppr, tpr):
        ropr = 0.27 * (ppr / (z * tpr))
        func = (
            -z
            + 1
            + (0.3265 - 1.0700 / tpr - 0.5339 / tpr**3 + 0.01569 / tpr**4 - 0.05165 / tpr**5) * ropr
            + (0.5475 - 0.7361 / tpr + 0.1844 / tpr**2) * ropr**2
            - 0.1056 * (-0.7361 / tpr + 0.1844 / tpr**2) * ropr**5
            + 0.6134 * (1 + 0.7210 * ropr**2) * (ropr**2 / tpr**3) * (np if not isinstance(ropr, torch.Tensor) else torch).exp(-0.7210 * ropr**2)
        )
        return func

    def __z_dranchuk(self, ppr: float, tpr: float) -> float:
        """
        Метод расчета z-фактора по корреляции Dranchuk

        Parameters
        ----------
        :param ppr - псевдоприведенное давление, доли
        :param tpr - псевдоприведенная температура, доли

        :return: z - фактор

        ref 1 Dranchuk, P.M. and Abou-Kassem, J.H.
        “Calculation of Z Factors for Natural
        Gases Using Equations of State.”
        Journal of Canadian Petroleum Technology. (July–September 1975) 34–36.
        -------
        """

        try:
            z_dranchuk = opt.brentq(self.__dak_func, a=0.1, b=8, args=(ppr, tpr))
        except ValueError:
            z_dranchuk = opt.newton(self.__dak_func, x0=1, args=(ppr, tpr), tol=1.48e-2)
        return z_dranchuk

    def __z_kareem(self, ppr: float, tpr: float) -> float:
        """
        Метод расчета z-фактора по корреляции Kareem с учетом границ применимости

        Parameters
        ----------
        :param ppr: псевдоприведенное давление, доли
        :param tpr: псевдоприведенная температура, доли

        :return: z - фактор
        """
        if 0.2 <= ppr <= 15 and 1.15 <= tpr <= 3:
            # Вызов метода расчета z-фактора по корреляции Kareem
            z = self.__z_kareem_(ppr, tpr)
        else:
            # Вызов метода расчета z-фактора по корреляции Dranchuk
            z = self.__z_dranchuk(ppr, tpr)
        return z

    @staticmethod
    def __z_standing(ppr: float, tpr: float) -> float:
        """
        Метод расчета z-фактора по корреляции Standing из Pipesim

        :param ppr: псевдоприведенное давление, доли
        :param tpr: псевдоприведенная температура, доли

        :return: z - фактор
        """
        a = 1.39 * (tpr - 0.92) ** 0.5 - 0.36 * tpr - 0.101
        b = (0.62 - 0.23 * tpr) * ppr
        c = (0.066 / (tpr - 0.86) - 0.037) * ppr**2
        d = (0.32 / (10 ** (9 * (tpr - 1)))) * ppr**6
        e = b + c + d
        f = 0.132 - 0.32 * (np if not isinstance(tpr, torch.Tensor) else torch).log10(tpr)
        g = 10 ** (0.3106 - 0.49 * tpr + 0.1824 * tpr**2)

        z = a + (1 - a) * (np if not isinstance(e, torch.Tensor) else torch).exp(-e) + f * ppr**g
        return z

    @staticmethod
    def __z_kareem_(ppr: float, tpr: float) -> float:
        """
        Метод расчета z-фактора по корреляции Kareem

        Parameters
        ----------
        based on  https://link.springer.com/article/10.1007/s13202-015-0209-3
        Kareem, L.A., Iwalewa, T.M. & Al-Marhoun, M.

        New explicit correlation for the compressibility factor
        of natural gas: linearized z-factor isotherms.
        J Petrol Explor Prod Technol 6, 481–492 (2016).
        https://doi.org/10.1007/s13202-015-0209-3

        :param ppr: псевдоприведенное давление, доли
        :param tpr: псевдоприведенная температура, доли

        :return: z - фактор
        -------
        """

        a = [
            0.317842,
            0.382216,
            -7.768354,
            14.290531,
            0.000002,
            -0.004693,
            0.096254,
            0.16672,
            0.96691,
            0.063069,
            -1.966847,
            21.0581,
            -27.0246,
            16.23,
            207.783,
            -488.161,
            176.29,
            1.88453,
            3.05921,
        ]

        t = 1 / tpr
        t2 = t**2
        t3 = t**3
        aa = a[0] * t * (np if not isinstance(t, torch.Tensor) else torch).exp(a[1] * (1 - t) ** 2) * ppr
        bb = a[2] * t + a[3] * t2 + a[4] * t**6 * ppr**6
        cc = a[8] + a[7] * t * ppr + a[6] * t2 * ppr**2 + a[5] * t3 * ppr**3
        dd = a[9] * t * (np if not isinstance(t, torch.Tensor) else torch).exp(a[10] * (1 - t) ** 2)
        ee = a[11] * t + a[12] * t2 + a[13] * t3
        ff = a[14] * t + a[15] * t2 + a[16] * t3
        gg = a[17] + a[18] * t

        dppr = dd * ppr
        y = dppr / ((1 + aa**2) / cc - aa**2 * bb / (cc**3))
        z = dppr * (1 + y + y**2 - y**3) / (dppr + ee * y**2 - ff * y**gg) / ((1 - y) ** 3)

        return z

    def calc_z(self, p: float, t: float, gamma_gas: float, **kwargs) -> float:
        """
        Метод расчета z-фактора, в котором в зависимости от указанного типа корреляции вызывается \
        соответствующий метод расчета

        Parameters
        ----------
        :param p: давление, Па
        :param t: температура, К
        :param gamma_gas: относительная плотность газа, доли,
        (относительно в-ха с плотностью 1.2217 кг/м3 при с.у.)

        :return: z - фактор
        -------
        """
        # Вызов расчета псевдокритических параметров и переход к псевдоприведенным значениям
        ppr = p / self.ppc(gamma_gas)
        tpr = t / self.tpc(gamma_gas)
        return self.z(ppr, tpr)

    @staticmethod
    def calc_gas_fvf(p: float, t: float, z: float, **kwargs) -> float:
        """
        Метод расчета объемного коэффициента газа,
        в котором в зависимости от указанного типа корреляции вызывается \
        соответствующий метод расчета

        Parameters
        ----------
        :param p: давление, Па
        :param t: температура, К
        :param z: коэффициент сжимаемости газа, 1/Па

        :return: объемный коэффициент газа, м3/м3
        -------
        """
        gas_fvf = t * z * 350.958 / p
        return gas_fvf

    @staticmethod
    def calc_gas_density(gamma_gas: float, bg: float, **kwargs) -> float:
        """
        Метод расчета плотности газа,
        в котором в зависимости от указанного типа корреляции вызывается \
        соответствующий метод расчета

        Parameters
        ----------
        :param gamma_gas: относительная плотность газа, доли,
        (относительно в-ха с плотностью 1.2217 кг/м3 при с.у.)
        :param bg: объемный коэффициент газа, м3/м3

        :return: плотность газа, кг/м3
        -------
        """
        m = 28.97 * gamma_gas
        gas_density = m / (24.04220577350111 * bg)
        return gas_density

    @staticmethod
    def __gas_viscosity_lee(t: float, gamma_gas: float, rho_gas: float) -> float:
        """
        Метод расчета вязкости газа по корреляции Lee

        Parameters
        ----------
        :param t: температура, К
        :param gamma_gas: относительная плотность газа, доли,
        (относительно в-ха с плотностью 1.2217 кг/м3 при с.у.)
        :param rho_gas: плотность газа при данном давлении температуре, кг/м3
        :return: вязкость газа, сПз
        -------
        """
        t_r = t * 1.8

        # Новая корреляция Lee
        # m = 28.9612403 * gamma_gas  # молярная масса газа
        # a = ((9.379 + 0.01607 * m) * t_r ** 1.5) / (209.2 + 19.26 * m + t_r)
        # b = 3.448 + 986.4 / t_r + 0.01009 * gamma_gas * 28.966
        # c = 2.447 - 0.2224 * b
        # Старая корреляция Lee как в Pipesim
        a = (7.77 + 0.183 * gamma_gas) * t_r**1.5 / (122.4 + 373.6 * gamma_gas + t_r)
        b = 2.57 + 1914.5 / t_r + 0.275 * gamma_gas
        c = 1.11 + 0.04 * b
        gas_viscosity = 10 ** (-4) * a * (np if not isinstance(b, torch.Tensor) else torch).exp(b * (rho_gas / 1000) ** c)
        return gas_viscosity

    def calc_gas_viscosity(self, t: float, gamma_gas: float, rho_gas: float, **kwargs) -> float:
        """
        Метод расчета вязкости газа, в котором в зависимости
        от указанного типа корреляции вызывается \
        соответствующий метод расчета

        Parameters
        ----------
        :param t: температура, К
        :param gamma_gas: относительная плотность газа, доли,
        (относительно в-ха с плотностью 1.2217 кг/м3 при с.у.)
        :param rho_gas: плотность газа, кг/м3

        :return: вязкость газа, сПз
        -------
        """
        return self.mu(t, gamma_gas, rho_gas)

    def calc_heat_capacity(self, p, t, gamma_gas, **kwargs):
        """
        Расчет удельной теплоемкости газа

        Parameters
        ----------
        :param p: давление, Па
        :param t: температура, К
        :param gamma_gas: относительная плотность газа, доли,
        (относительно в-ха с плотностью 1.2217 кг/м3 при с.у.)

        Returns
        -------
        Удельная теплоемкость газа, Дж/(кг*К)
        """
        return self.hc_gas(p=p, t=t, gamma_gas=gamma_gas)

    @staticmethod
    def __calc_heat_capacity_const(**kwargs):
        """
        Выдача постоянного значения удельной теплоемкости газа = 2302.733, Дж/(кг*К)

        :return:
        """
        return 2302.733

    @staticmethod
    def __calc_heat_capacity_mahmood(p: float, t: float, gamma_gas: float) -> float:
        """
        Расчет удельной теплоемкости газа

        Parameters
        ----------
        :param p: давление, Па
        :param t: температура, К
        :param gamma_gas: относительная плотность газа, доли,
        (относительно в-ха с плотностью 1.2217 кг/м3 при с.у.)

        Returns
        -------
        Удельная теплоемкость газа, Дж/(кг*К)

        Ref:
        Mahmood Moshfeghian Petroskills
        """
        t_c = t - 273.15
        p_mpa = p * 10 ** (-6)
        a = 0.9
        b = 1.014
        c = -0.7
        d = 2.170
        e = 1.015
        f = 0.0214
        return (a * b**t_c * t_c**c + d * e**p_mpa * p_mpa**f) * (gamma_gas / 0.6) ** 0.025 * 1000
    
    def calc_gas_compressibility(self, p: float, t: float, z: float, **kwargs):
        """
        Расчет удельной теплоемкости газа

        Parameters
        ----------
        :param p: давление, Па
        :param t: температура, К
        :param gamma_gas: относительная плотность газа, доли,
        (относительно в-ха с плотностью 1.2217 кг/м3 при с.у.)

        Returns
        -------
        Удельная теплоемкость газа, Дж/(кг*К)
        """
        bg1 = self.calc_gas_fvf(p, t, z)
        bg2 = self.calc_gas_fvf(p + 1, t, z)
        return abs(bg2 - bg1) / bg1
