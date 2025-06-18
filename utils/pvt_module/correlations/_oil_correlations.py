"""
Модуль, для описания корреляций для расчета свойств нефти
"""
import numpy as np
import torch

import service.constants as const


class OilCorrelations:
    """
    Класс, включающий функции для расчета нефтяных свойств от давления и температуры
    """

    __slots__ = [
        "pb",
        "bo_below",
        "rs",
        "bo_above",
        "compro",
        "mud",
        "mus",
        "muu",
        "hc_oil",
        "st_oil_gas",
    ]

    def __init__(self, oil_correlations):
        """
        :param oil_correlations: словарь с набором корреляций для каждого свойства
        """
        if oil_correlations["pb"] == "standing":
            self.pb = self.__pb_standing
        else:
            raise ValueError(
                f"Корреляция {oil_correlations['pb']} для давления "
                f"насыщения пока не реализована."
                f"Используйте другую корреляцию",
            )

        if oil_correlations["rs"] == "standing":
            self.rs = self.__rs_standing
        else:
            raise ValueError(
                f"Корреляция {oil_correlations['rs']} для газосодержания"
                f" пока не реализована."
                f"Используйте другую корреляцию",
            )

        if oil_correlations["b"] == "standing":
            self.bo_below = self.__oil_fvf_standing_below
        else:
            raise ValueError(
                f"Корреляция {oil_correlations['b']} для объемного "
                f"коэффициента нефти пока не реализована. "
                f"Используйте другую корреляцию",
            )
        self.bo_above = self.__oil_fvf_vasquez_above

        if oil_correlations["compr"] == "vasquez":
            self.compro = self.__oil_compr_vasquez
        else:
            raise ValueError(
                f"Корреляция {oil_correlations['compr']} для сжимаемости"
                f" нефти пока не реализована."
                f"Используйте другую корреляцию",
            )

        if oil_correlations["mud"] == "beggs":
            self.mud = self.__oil_deadviscosity_beggs
        else:
            raise ValueError(
                f"Корреляция {oil_correlations['mud']} для вязкости "
                f"дегазированной нефти пока не реализована."
                f"Используйте другую корреляцию",
            )

        if oil_correlations["mus"] == "beggs":
            self.mus = self.__oil_liveviscosity_beggs
        else:
            raise ValueError(
                f"Корреляция {oil_correlations['mus']} для вязкости "
                f"насыщенной нефти пока не реализована."
                f"Используйте другую корреляцию",
            )

        self.muu = self._oil_viscosity_vasquez_beggs

        if oil_correlations["hc"] == "wright":
            self.hc_oil = self.__calc_heat_capacity_wright
        elif oil_correlations["hc"] == "const":
            self.hc_oil = self.__calc_heat_capacity_const
        else:
            raise ValueError(
                f"Корреляция {oil_correlations['hc']} для удельной теплоемкости "
                f"нефти пока не реализована."
                f"Используйте другую корреляцию",
            )

        if oil_correlations["st_oil_gas"] == "baker":
            self.st_oil_gas = self.__calc_st_oil_gas_baker
        else:
            raise ValueError(
                f"Корреляция {oil_correlations['st_oil_gas']} для поверхностного "
                f"натяжение на границе нефть-газ пока не реализована."
                f"Используйте другую корреляцию",
            )

    @staticmethod
    def __pb_standing(rsb, gamma_oil, gamma_gas, t):
        """
        Метод расчета давления насыщения по корреляции Standing

        Parameters
        ----------
        :param rsb: газосодержание при давлении насыщения, м3/м3
        :param gamma_oil: относительная плотность нефти, доли,
        (относительно воды с плотностью 1000 кг/м3 при с.у.)
        :param gamma_gas: относительная плотность газа, доли,
        (относительно в-ха с плотностью 1.2217 кг/м3 при с.у.)
        :param t: температура, К

        :return: давление насыщения, Па
        -------
        """
        # Нужно будет понять почему это значение минимально
        rsb_min = 1.8
        rsb_old = rsb

        rsb = max(rsb_min, rsb)

        # Правильный вариант без округлений ! Эталон :)
        yg = 1.2254503 + 0.001638 * t - 1.76875 / gamma_oil
        pb_standing = 519666.7519706273 * ((rsb / gamma_gas) ** 0.83) * (10**yg)

        if rsb_old < rsb_min:
            pb_standing = (pb_standing - 101325) * rsb_old / rsb_min + 101325

        return pb_standing

    def calc_pb(self, t, rsb, gamma_oil, gamma_gas, **kwargs):
        """
        Метод расчета давления насыщения, в котором в зависимости
        от указанного типа корреляции вызывается \
        соответствующий метод расчета

        Parameters
        ----------
        :param t: температура, К
        :param rsb: газосодержание при давлении насыщения, м3/м3
        :param gamma_oil: относительная плотность нефти, доли,
        (относительно воды с плотностью 1000 кг/м3 при с.у.)
        :param gamma_gas: относительная плотность газа, доли,
        (относительно в-ха с плотностью 1.2217 кг/м3 при с.у.)

        :return: давление насыщения, Па
        -------
        """
        return self.pb(rsb, gamma_oil, gamma_gas, t)

    @staticmethod
    def __rs_standing(p, t, gamma_oil, gamma_gas):
        """
        Метод расчета газосодержания по корреляции Standing

        Parameters
        ----------
        :param p: давление насыщения, Па
        :param t: температура, К
        :param gamma_oil: относительная плотность нефти, доли),
        (относительно воды с плотностью 1000 кг/м3 при с.у.)
        :param gamma_gas: относительная плотность газа, доли,
        (относительно в-ха с плотностью 1.2217 кг/м3 при с.у.)

        :return: газосодержание, м3/м3
        -------
        """
        # Правильный вариант без округлений ! Эталон :)
        yg = 1.2254503 + 0.001638 * t - 1.76875 / gamma_oil
        rs = gamma_gas * (1.9243101395421235e-06 * p / 10**yg) ** 1.2048192771084338
        return rs

    def calc_rs(self, p, t, gamma_oil, gamma_gas, **kwargs):
        """
        Метод расчета газосодержания, в котором в зависимости
        от указанного типа корреляции вызывается \
        соответствующий метод расчета

        Parameters
        ----------
        :param p: давление, Па
        :param t: температура, К
        :param gamma_oil: относительная плотность нефти, доли,
        (относительно воды с плотностью 1000 кг/м3 при с.у.)
        :param gamma_gas: относительная плотность газа, доли,
        (относительно в-ха с плотностью 1.2217 кг/м3 при с.у.)

        :return: газосодержание, м3/м3
        -------
        """
        return self.rs(p, t, gamma_oil, gamma_gas)

    @staticmethod
    def __oil_fvf_standing_below(rs, gamma_gas, gamma_oil, t):
        """
        Метод расчета объемного коэффициента нефти по корреляции Standing

        Parameters
        ----------
        :param rs: газосодержание, м3/м3
        :param gamma_oil: относительная плотность нефти, доли,
        (относительно воды с плотностью 1000 кг/м3 при с.у.)
        :param gamma_gas: относительная плотность газа, доли,
        (относительно в-ха с плотностью 1.2217 кг/м3 при с.у.)
        :param t: температура, К

        :return: объемный коэффициент нефти, м3/м3
        -------
        """

        bo = 0.972 + 0.000147 * (
            (5.614583333333334 * rs * ((gamma_gas / gamma_oil) ** 0.5) + 2.25 * t - 574.5875) ** 1.175
        )
        return bo

    @staticmethod
    def __oil_fvf_vasquez_above(p, compr, pb, bob):
        """
        Метод расчета объемного коэффициента нефти по корреляции Vasquez
        при давлении выше давления насыщения

        Parameters
        ----------
        :param p: давление, Па
        :param compr: сжимаемость нефти, 1/Па
        :param pb: давление насыщения, Па
        :param bob: объемный коэффициент при давлении насыщения, безразм.

        :return: объемный коэффициент нефти, м3/м3
        -------
        """
        oil_fvf_vasquez_above = bob * (np if not (isinstance(pb, torch.Tensor) or isinstance(p, torch.Tensor) or isinstance(compr, torch.Tensor)) else torch).exp(compr * (p - pb))
        return oil_fvf_vasquez_above

    def calc_oil_fvf(self, p, t, rs, gamma_oil, gamma_gas, compr, pb, bob=None, **kwargs):
        """
        Метод расчета объемного коэффициента нефти, в котором в зависимости
        от указанного типа корреляции вызывается \
        соответствующий метод расчета

        Parameters
        ----------
        :param p: давление, Па
        :param t: температура, К
        :param rs: газосодержание, м3/м3
        :param gamma_oil: относительная плотность нефти, доли,
        (относительно воды с плотностью 1000 кг/м3 при с.у.)
        :param gamma_gas: относительная плотность газа, доли,
        (относительно в-ха с плотностью 1.2217 кг/м3 при с.у.)
        :param compr: сжимаемость нефти, 1/Па
        :param pb: давление насыщения, Па
        :param bob: объемный коэффициент нефти при давлении насыщения, м3/м3

        :return: объемный коэффициент нефти, м3/м3
        -------
        """
        if p <= pb or (np if not isinstance(p, torch.Tensor) else torch).isnan(p):
            bo = self.bo_below(rs, gamma_gas, gamma_oil, t)
        else:
            # Вызов метода расчета объемного коэффициента нефти по Vasquez
            # при давлении выше давления насыщения
            bo = self.bo_above(p, compr, pb, bob)
        return bo

    @staticmethod
    def __oil_compr_vasquez(t, gamma_oil, gamma_gas, rsb, p):
        """
        Метод расчета коэффициента сжимаемости нефти по корреляции Vasquez

        Parameters
        ----------
        :param t: температура, К
        :param gamma_oil: относительная плотность нефти, доли,
        (относительно воды с плотностью 1000 кг/м3 при с.у.)
        :param gamma_gas: относительная плотность газа, доли,
        (относительно в-ха с плотностью 1.2217 кг/м3 при с.у.)
        :param rsb: газосодержание при давлении насыщении, м3/м3
        :param p: давление, Па

        :return: коэффициент сжимаемости нефти, 1/Па
        -------
        """
        return max(
            (28.07291666666667 * rsb + 30.96 * (t - 273) - 1180 * gamma_gas + 1784.3149999999998 / gamma_oil - 2540.8) / 100000 / p,
            2 * 10**-12,
        )

    def calc_oil_compressibility(self, t, gamma_oil, gamma_gas, rsb, p, **kwargs):
        """
        Метод расчета коэффициента сжимаемости нефти, в котором в зависимости
        от указанного типа корреляции вызывается \
        соответствующий метод расчета

        Parameters
        ----------
        :param t: температура, К
        :param gamma_oil: относительная плотность нефти, доли,
        (относительно воды с плотностью 1000 кг/м3 при с.у.)
        :param gamma_gas: относительная плотность газа, доли,
        (относительно в-ха с плотностью 1.2217 кг/м3 при с.у.)
        :param rsb: газосодержание при давлении насыщения, м3/м3
        :param p: давление, Па

        :return: коэффициент сжимаемости нефти, 1/Па
        -------
        """
        return self.compro(t, gamma_oil, gamma_gas, rsb, p)

    @staticmethod
    def calc_oil_density(rs, bo, gamma_oil, gamma_gas, **kwargs):
        """
        Метод расчета плотности нефти, в котором в зависимости
        от указанного типа корреляции вызывается \
        соответствующий метод расчета

        Parameters
        ----------
        :param rs: газосодержание, м3/м3
        :param bo: объемный коэффициент нефти, м3/м3
        :param gamma_oil: относительная плотность нефти, доли,
        (относительно воды с плотностью 1000 кг/м3 при с.у.)
        :param gamma_gas: относительная плотность газа, доли,
        (относительно в-ха с плотностью 1.2217 кг/м3 при с.у.)

        :return: плотность нефти, кг/м3
        -------
        """
        oil_density = 1000 * (gamma_oil + rs * gamma_gas * 1.2217 / 1000) / bo
        return oil_density

    @staticmethod
    def __oil_deadviscosity_beggs(gamma_oil, t):
        """
        Метод расчета вязкости дегазированной нефти по корреляции Beggs

        Parameters
        ----------
        :param gamma_oil: относительная плотность нефти, доли,
        (относительно воды с плотностью 1000 кг/м3 при с.у.)
        :param t: температура, К

        :return: вязкость дегазированной нефти, сПз
        -------
        """
        # Ограничение плотности нефти = 58 API для корреляции Beggs and Robinson
        gamma_oil = min((141.5 / gamma_oil - 131.5), 58)

        # Ограничение температуры = 295 F для корреляции Beggs and Robinson
        t = min(((t - 273.15) * 1.8 + 32), 295)

        if t < 70:
            # Корректировка вязкости дегазированной нефти для температуры ниже 70 F
            # Расчет вязкости дегазированной нефти для 70 F
            oil_deadviscosity_beggs_70 = (10 ** ((10 ** (3.0324 - 0.02023 * gamma_oil)) * (70 ** (-1.163)))) - 1
            # Расчет вязкости дегазированной нефти для 80 F
            oil_deadviscosity_beggs_80 = (10 ** ((10 ** (3.0324 - 0.02023 * gamma_oil)) * (80 ** (-1.163)))) - 1
            # Экстраполяция вязкости дегазированной нефти по двум точкам
            c = np.log10(oil_deadviscosity_beggs_70 / oil_deadviscosity_beggs_80) / np.log10(80 / 70)
            b = oil_deadviscosity_beggs_70 * 70**c
            oil_deadviscosity_beggs = 10 ** ((np if not isinstance(b, torch.Tensor) else torch).log10(b) - c * (np if not isinstance(t, torch.Tensor) else torch).log10(t))
        else:
            x = (10 ** (3.0324 - 0.02023 * gamma_oil)) * (t ** (-1.163))
            oil_deadviscosity_beggs = (10**x) - 1
        return oil_deadviscosity_beggs

    @staticmethod
    def __oil_liveviscosity_beggs(oil_deadvisc, rs):
        """
        Метод расчета вязкости нефти, насыщенной газом, по корреляции Beggs

        Parameters
        ----------
        :param oil_deadvisc: вязкость дегазированной нефти, сПз
        :param rs: газосодержание, (м3/м3)

        :return: вязкость, насыщенной газом нефти, сПз
        -------
        """
        # Конвертация газосодержания в куб. футы/баррель
        rs_new = rs / 0.17810760667903522

        a = 10.715 * (rs_new + 100) ** (-0.515)
        b = 5.44 * (rs_new + 150) ** (-0.338)
        oil_liveviscosity_beggs = a * oil_deadvisc**b
        return oil_liveviscosity_beggs

    @staticmethod
    def _oil_viscosity_vasquez_beggs(oil_liveviscosity, p, pb):
        """
        Метод расчета вязкости нефти по корреляции Beggs

        Parameters
        ----------
        :param oil_liveviscosity: вязкость нефти, насыщенной газом, сПз
        :param p: давление, Па
        :param pb: давление насыщения, Па

        :return: вязкость нефти, сПз
        -------
        """
        m = 957 * (p * 10 ** (-6)) ** 1.187 * (np if not isinstance(p, torch.Tensor) else torch).exp(-11.513 - 1.302e-08 * p)
        oil_viscosity_vasquez_beggs = oil_liveviscosity * (p / pb) ** m

        return oil_viscosity_vasquez_beggs

    def calc_oil_viscosity(self, p, t, pb, gamma_oil, rs, calibr_mu, **kwargs):
        """
        Метод расчета вязкости нефти, в котором в зависимости
        от указанного типа корреляции вызывается \
        соответствующий метод расчета

        Parameters
        ----------
        :param p: давление, Па
        :param t: температура, К
        :param pb: давление насыщения, Па
        :param gamma_oil: относительная плотность нефти, доли,
        (относительно воды с плотностью 1000 кг/м3 при с.у.)
        :param rs: газосодержание, м3/м3
        :param calibr_mu: калибровочный коэффициент для вязкости газонасыщенной нефти, д.ед.

        :return: вязкость нефти, сПз
        -------
        """
        # Вызов метода расчета вязкости дегазированной нефти по корреляции Beggs
        oil_deadvisc = self.mud(gamma_oil, t)

        if rs != 0:
            oil_viscosity = self.mus(oil_deadvisc, rs)
        else:
            oil_viscosity = oil_deadvisc

        oil_viscosity = (oil_viscosity * oil_deadvisc) / ((1 - calibr_mu) * oil_viscosity + calibr_mu * oil_deadvisc)

        if p > pb:
            oil_viscosity = self.muu(oil_viscosity, p, pb)

        return oil_viscosity

    @staticmethod
    def __calc_heat_capacity_const(**kwargs):
        """
        Выдача постоянного значения удельной теплоемкости нефти = 1884.054 (Дж/(кг*К))

        :return: Удельная теплоемкость нефти, Дж/(кг*К)
        """
        return 1884.054

    def calc_heat_capacity(self, t, gamma_oil, **kwargs):
        """
        Расчет удельной теплоемкости нефти

        :param t: температура, К
        :param gamma_oil: относительная плотность нефти, доли,
        (относительно воды с плотностью 1000 кг/м3 при с.у.)

        :return: Удельная теплоемкость нефти, Дж/(кг*К)
        """
        return self.hc_oil(t=t, gamma_oil=gamma_oil)

    @staticmethod
    def __calc_heat_capacity_wright(t, gamma_oil):
        """
        Расчет удельной теплоемкости нефти по корреляции Wes Wright

        Parameters
        ----------
        :param t: температура, К
        :param gamma_oil: относительная плотность нефти, доли,
        (относительно воды с плотностью 1000 кг/м3 при с.у.)


        :return: Удельная теплоемкость нефти, Дж/(кг*К)

        Ref
        ___
        Wes Wright, Petroskills
        """
        t_c = t - 273.15
        return ((2 * t_c * 10 ** (-3) - 1.429) * gamma_oil + (2.67 * 10 ** (-3)) * t_c + 3.049) * 1000

    def calc_st_oil_gas(self, t, gamma_oil, rs, **kwargs):
        """
        Метод расчета коэффициента поверхностного натяжения на границе газ-нефть
        :return:
        """
        return self.st_oil_gas(t, gamma_oil, rs)

    @staticmethod
    def __calc_st_oil_gas_baker(t, gamma_oil, rs):
        """
        Метод расчета коэффициента поверхностного натяжения на границе
        газ-нефть по корреляции Baker and Swerdloff

        Parameters
        ----------
        :param t: температура, К
        :param gamma_oil: относительная плотность нефти, доли,
        (относительно воды с плотностью 1000 кг/м3 при с.у.)
        :param rs: газосодержание, м3/м3

        :return: st (superficial tension) - поверхностное натяжение, Н/м
        -------
        """
        # Конвертация температуры в градусы Фаренгейта
        t = (t - 273.15) * 1.8 + 32

        rs = rs * 5.614583333333333
        sto = (
            (1.17013 - 0.001694 * t)
            * (38.085 - 0.259 * (141.5 / gamma_oil - 131.5))
            * (0.056379 + 0.94362 * (np if not isinstance(rs, torch.Tensor) else torch).exp(-0.0038491 * rs))
        )
        return sto / 1000
