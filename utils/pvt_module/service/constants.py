OIL_CORRS = {
    "pb": "standing",
    "rs": "standing",
    "b": "standing",
    "mud": "beggs",
    "mus": "beggs",
    "compr": "vasquez",
    "hc": "wright",
    "st_oil_gas": "baker",
}
WAT_CORRS = {
    "b": "mccain",
    "mu": "mccain",
    "rho": "standing",
    "hc": "const",
    "st_wat_gas": "katz",
}
GAS_CORRS = {
    "ppc": "standing",
    "tpc": "standing",
    "z": "standing",
    "mu": "lee",
    "hc": "mahmood",
}

ATM = 101325

PB_MAX = 137895145.863367

ALL_PROPERTIES = {
    "bw": ["wat_corrs", "calc_water_fvf"],
    "rho_wat": ["wat_corrs", "calc_water_density"],
    "muw": ["wat_corrs", "calc_water_viscosity"],
    "hc_wat": ["wat_corrs", "calc_heat_capacity"],
    "salinity": ["wat_corrs", "calc_salinity"],
    "z": ["gas_corrs", "calc_z"],
    "bg": ["gas_corrs", "calc_gas_fvf"],
    "rho_gas": ["gas_corrs", "calc_gas_density"],
    "mug": ["gas_corrs", "calc_gas_viscosity"],
    "hc_gas": ["gas_corrs", "calc_heat_capacity"],
    "pb": ["oil_corrs", "calc_pb"],
    "rs": ["oil_corrs", "calc_rs"],
    "compro": ["oil_corrs", "calc_oil_compressibility"],
    "bo": ["oil_corrs", "calc_oil_fvf"],
    "rho_oil": ["oil_corrs", "calc_oil_density"],
    "muo": ["oil_corrs", "calc_oil_viscosity"],
    "hc_oil": ["oil_corrs", "calc_heat_capacity"],
    "st_wat_gas": ["wat_corrs", "calc_st_wat_gas"],
    "st_oil_gas": ["oil_corrs", "calc_st_oil_gas"],
    "comprw": ["wat_corrs", "calc_water_compressibility"],
    "comprg": ["gas_corrs", "calc_gas_compressibility"]
}

PSI = 0.00014503773773020924