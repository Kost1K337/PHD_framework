from astropy import units as u

ATM = u.def_unit("atm", u.Pa * 101325)
CUB = u.m**3 / u.day
CUB_BAR = CUB / u.bar
CUB_ATM = CUB / ATM
TON_DAY = u.tonne / u.day
M3_M3 = u.m**3 / u.m**3
CUB_CUB = CUB / CUB
KG_M3 = u.kg / u.m**3
TON_DAY = u.tonne / u.day
KG_SEC = u.kg / u.s
KG_DAY = u.kg / u.day
PA_KG_S = u.Pa / KG_SEC
ATM_KG_S = ATM / KG_SEC
KG_KG = u.kg / u.kg
M3_SEC = u.m**3 / u.s
PA_X_S = u.Pa * u.s
KG_M3_KG_M3 = (u.kg / u.m**3) / (u.kg / u.m**3)
PA_M = u.Pa / u.m
PA = u.Pa
BAR_M = u.bar / u.m

A = 1 / (u.day * u.bar)
B = u.m**3 / (u.day * u.bar)

DEG_CELS = u.deg_C
DEG_KELV = u.K # Переводить через DEG_CELS.to(DEG_KELV, u.temperature()))
WATT = u.kg * u.m**2 / u.s**3 # определение ватта
W_M2_DEG_KELV = WATT / u.m**2 / DEG_KELV # для heat_transfer_coefficient

J = u.kg*u.m**2/u.s**2
J_KG_DEG_K = J/u.kg/DEG_KELV # для heat_capacity

UNITS = {
    "qoil": CUB,
    "qwat": CUB,
    "qgas": CUB,
    "gor": CUB_CUB,
    "wct": CUB_CUB,
    "qliq": CUB,
    "pres": u.bar
}
