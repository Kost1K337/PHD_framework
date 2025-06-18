"""
Модуль, для описания корреляций для расчета свойств воды
"""
import service.constants as const


class WaterCorrelations:
    """
    Класс, включающий функции для расчета свойств воды от давления и температуры
    """

    __slots__ = ["bw", "rho_wat", "muw", "hc_wat", "st_wat_gas"]

    def __init__(self, water_correlations):
        """
        :param water_correlations: словарь с набором корреляций для каждого свойства
        """
        if water_correlations["b"] == "mccain":
            self.bw = self.__water_fvf_mccain
        else:
            raise ValueError(
                f"Корреляция {water_correlations['b']} для объемного"
                f"коэффициента воды пока не реализована."
                f"Используйте другую корреляцию",
            )

        if water_correlations["rho"] == "standing":
            self.rho_wat = self.__water_density_standing
        else:
            raise ValueError(
                f"Корреляция {water_correlations['rho']} для плотности воды"
                f" пока не реализована. Используйте другую корреляцию",
            )

        if water_correlations["mu"] == "mccain":
            self.muw = self.__water_viscosity_mccain
        else:
            raise ValueError(
                f"Корреляция {water_correlations['mu']} для вязкости воды"
                f" пока не реализована. Используете другую корреляцию",
            )

        if water_correlations["hc"] == "const":
            self.hc_wat = self.__calc_heat_capacity_const
        else:
            raise ValueError(
                f"Корреляция {water_correlations['hc']} для удельной теплоемкости воды"
                f" пока не реализована. Используете другую корреляцию",
            )

        if water_correlations["st_wat_gas"] == "katz":
            self.st_wat_gas = self.__calc_st_wat_gas_katz
        else:
            raise ValueError(
                f"Корреляция {water_correlations['st_wat_gas']} для поверхностного"
                f"натяжения на границе вода-газ пока не реализована. "
                f"Используете другую корреляцию",
            )

    @staticmethod
    def __water_density_standing(gamma_wat, bw, **kwargs):
        """
        Метод расчета плотности воды

        Parameters
        ----------
        :param gamma_wat: относительная плотность воды, доли,
        (относительно воды с плотностью 1000 кг/м3)
        :param bw: объемный коэффициент воды, м3/м3

        :return: плотность воды, кг/м3
        -------
        """

        water_density = 1000 * gamma_wat / bw
        return water_density

    def calc_water_density(self, t, p, gamma_wat, bw, salinity, **kwargs):
        """
        Метод расчета плотности воды, в котором
        в зависимости от указанного типа корреляции вызывается \
        соответствующий метод расчета

        Parameters
        ----------
        :param t: температура, К
        :param p: давление, Па
        :param gamma_wat: относительная плотность воды, доли,
        (относительно воды с плотностью 1000 кг/м3 при с.у.)
        :param bw: объемный коэффициент воды, м3/м3
        :param salinity: минерализация воды, ppm
        :return: плотность воды, кг/м3
        -------
        """
        return self.rho_wat(gamma_wat=gamma_wat, t=t, p=p, bw=bw, salinity=salinity)

    @staticmethod
    def __water_fvf_mccain(t, p):
        """
        Метод расчета объемного коэффициента воды

        Parameters
        ----------
        :param t: температура, К
        :param p: давление, Па

        :return: объемный коэффициент воды, м3/м3
        -------
        """
        # Конвертация температуры в градусы Фаренгейта
        t = (t - 273.15) * 1.8 + 32
        # Конвертация давления МПа в psi
        p = p * const.PSI

        dvwp = (
            -1.95301 * (10 ** (-9)) * p * t
            - 1.72834 * (10 ** (-13)) * (p**2) * t
            - 3.58922 * (10 ** (-7)) * p
            - 2.25341 * (10 ** (-10)) * (p**2)
        )
        dvwt = -1.0001 * (10 ** (-2)) + 1.33391 * (10 ** (-4)) * t + 5.50654 * (10 ** (-7)) * (t**2)
        water_fvf_vba = (1 + dvwp) * (1 + dvwt)
        return water_fvf_vba

    def calc_water_fvf(self, t, p, **kwargs):
        """
        Метод расчета объемного коэффициента воды,
        в котором в зависимости от указанного типа корреляции вызывается \
        соответствующий метод расчета

        Parameters
        ----------
        :param t: температура, К
        :param p: давление, Па

        :return: объемный коэффициент воды, м3/м3
        -------
        """
        return self.bw(t, p)

    @staticmethod
    def __water_compressibility_kriel(t, p, salinity, **kwargs):
        """
        Метод расчета сжимаемости воды по методике Kriel

        Parameters
        ----------
        :param t: температура, К
        :param p: давление, Па
        :param salinity: минерализация воды, ppm

        :return: сжимаемость воды, 1/Па
        -------
        """

        # Конвертация температуры в градусы Фаренгейта
        t = (t - 273.15) * 1.8 + 32

        # Конвертация давления МПа в psi
        p = p * const.PSI

        water_compr = const.PSI / (7.033 * p + 0.5415 * salinity - 537 * t + 403300)
        return water_compr

    @staticmethod
    def calc_salinity(gamma_wat, **kwargs):
        """
        Функция для расчета солености через объемный коэффициент воды

        Parameters
        ----------
        :param gamma_wat: относительная плотность воды, доли,
        (относительно воды с плотностью 1000 кг/м3 при с.у.)

        :return: соленость, (ppm)
        -------
        """
        salinity = (624.711071129603 * gamma_wat / 0.0160185 - 20192.9595437054) ** 0.5 - 137.000074965329
        return salinity * 10000

    def calc_water_compressibility(self, t, p, salinity, **kwargs):
        """
        Метод расчета сжимаемости воды,
        в котором в зависимости от указанного типа корреляции вызывается \
        соответствующий метод расчета

        Parameters
        ----------
        :param t: температура, К
        :param p: давление, Па
        :param salinity: минерализация воды, ppm

        :return: коэффициент сжимаемости воды, 1/Па
        -------
        """
        return self.__water_compressibility_kriel(t, p, salinity)

    @staticmethod
    def __water_viscosity_mccain(t, p, salinity):
        """
        Метод расчета вязкости воды по корреляции McCain

        Parameters
        ----------
        :param t: температура, К
        :param p: давление, Па
        :param salinity: минерализация воды, ppm

        :return: вязкость воды, сПз
        -------
        """
        # Конвертация солености в %
        salinity = salinity / 10000

        # Конвертация давления МПа в psi
        p = p * const.PSI

        # Конвертация температуры в градусы Фаренгейта
        t = (t - 273.15) * 1.8 + 32

        a = 109.574 - (8.40564 * salinity) + (0.313314 * (salinity**2)) + (0.00872213 * (salinity**3))
        b = (
            -1.12166
            + 0.0263951 * salinity
            - 0.000679461 * salinity**2
            - 5.47119 * 10 ** (-5) * salinity**3
            + 1.55586 * 10 ** (-6) * salinity**4
        )
        visc = a * t**b
        water_viscosity = visc * (0.9994 + 4.0295 * (10 ** (-5)) * p + 3.1062 * (10 ** (-9)) * (p**2))
        return water_viscosity

    def calc_water_viscosity(self, t, p, salinity, **kwargs):
        """
        Метод расчета вязкости воды,
        в котором в зависимости от указанного типа корреляции вызывается \
        соответствующий метод расчета

        Parameters
        ----------
        :param t: температура, К
        :param p: давление, Па
        :param salinity: минерализация воды, ppm

        :return: вязкость воды, сПз
        -------
        """

        return self.muw(t=t, p=p, salinity=salinity)

    def calc_heat_capacity(self, t, **kwargs):
        """
        Расчет удельной теплоемкости воды

        Parameters
        ----------
        :param t: температура, К

        Returns
        -------
        Удельная теплоемкость воды, Дж/(кг*К)
        """
        return self.hc_wat(t=t)

    @staticmethod
    def __calc_heat_capacity_const(**kwargs):
        """
        Выдача постоянного значения удельной теплоемкости воды = 4186.787 Дж/(кг*К)

        Returns
        -------
        Удельная теплоемкость воды, Дж/(кг*К)
        """
        return 4186.787

    def calc_st_wat_gas(self, p, t, **kwargs):
        """
        Метод расчета коэффициента поверхностного натяжения на границе газ-вода
        """
        return self.st_wat_gas(p, t)

    @staticmethod
    def __calc_st_wat_gas_katz(p, t):
        """
        Метод расчета коэффициента поверхностного натяжения на границе газ-вода по методу Katz

        Parameters
        ----------
        :param p: давление, Па
        :param t: температура, К

        :return: st (superficial tension) - поверхностное натяжение, Н/м
        -------
        """
        # Конвертация температуры в градусы Фаренгейта
        t = (t - 273.15) * 1.8 + 32

        # Конвертация давления в psi
        p_psi = p * const.PSI

        # Поставим пока как в Pipesim для проверки, вообще говоря у них более старые формулы
        stw = 70 - 0.1 * (t - 74) - 0.002 * p_psi

        return max(stw / 1000, 0.00001)
