import torch as tc
import torch.nn as nn
import numpy as np

class IProxyModel(nn.Module):
    well_count: int
    dt: np.float64

    def __init__(self, well_count: int, dt: np.float64):
        super().__init__()
        self.dt = dt
        self.well_count = well_count       

    def hidden_dim(self) -> int:
        pass 

    def start(self) -> tc.Tensor:
        pass

    def step(self, input: tc.Tensor, param_input: tc.Tensor) -> tc.Tensor:
        pass

    def view(self, input: tc.Tensor, param_input: tc.Tensor) -> tc.Tensor:
        pass

    def veiw_weights(self, input: tc.Tensor, param_input: tc.Tensor) -> tc.Tensor:
        pass

    def save(self, file_name):
        with open(file_name, 'w') as file:
            for name, param in self.named_parameters():
                file.write(f'name: {name}\nvalue: {param.data}\n\n')

class ProxyModelBase(IProxyModel):
    def __init__(self, well_count, dt):
        super().__init__(well_count, dt)

        perm = tc.rand((well_count, well_count))
        perm = (perm + tc.transpose(perm, 0, 1))

        self.permeability = nn.Parameter(perm)
        self.compressibility_o = nn.Parameter(tc.rand(well_count))
        self.compressibility_w = nn.Parameter(tc.rand(well_count))
        self.volume = nn.Parameter(tc.rand(well_count))

        self.permeability_rel_o = nn.Parameter(tc.ones(1))

        self.perm_aqu = nn.Parameter(tc.rand(well_count))
        self.J_aqu = nn.Parameter(tc.rand(1))
        self.bhp_aqu = nn.Parameter(tc.rand(1))

        self.pressure_init = nn.Parameter(tc.rand(well_count))
        self.saturation_init = nn.Parameter(tc.rand(well_count))

        self.register_parameter("permeability", self.permeability)
        self.register_parameter("compressibility_o", self.compressibility_o)
        self.register_parameter("compressibility_w", self.compressibility_w)
        self.register_parameter("volume", self.volume)

        self.register_parameter("permeability_rel_o", self.permeability_rel_o)

        self.register_parameter("perm_aqu", self.perm_aqu)
        self.register_parameter("J_aqu", self.J_aqu)
        self.register_parameter("bhp_aqu", self.bhp_aqu)

        self.register_parameter("pressure_init", self.pressure_init)
        self.register_parameter("saturation_init", self.saturation_init)

    def init_params(self, prod_hist: np.ndarray, sat_hist: np.ndarray, bhp_hist: np.ndarray, times: np.ndarray):
        time_steps = prod_hist.shape[0]
        well_count = prod_hist.shape[1]   

        self.time_steps = time_steps
        self.well_count = well_count     

        vol_mult = time_steps
        compr_mult = time_steps        

        self.vol_mult = vol_mult
        self.compr_mult = compr_mult

        self.dt = (times[-1] - times[0]) / (time_steps - 1)

        q0 = prod_hist[:-1]
        q1 = prod_hist[1:]
        s0 = sat_hist[:-1]
        s1 = sat_hist[1:]

        t0 = times[:-1]
        t1 = times[1:]
        dts = t1 - t0

        dob_o = ((q0 * (1.0 - s0) + q1 * (1.0 - s1)) * dts[:,None]).sum() * 0.5

        St = np.full(well_count, time_steps)

        Sq = prod_hist.sum(axis=0) / St
        Sb = bhp_hist.sum(axis=0) / St

        #Sb2 = (bhp_hist * bhp_hist).sum(axis=0) / St
        Sqb = (prod_hist * bhp_hist).sum(axis=0) / St
        Sq2 = (prod_hist * prod_hist).sum(axis=0) / St        

        s0 = sat_hist[0].mean()
        vol_tot = dob_o / (1.0 - s0 + 0.001)

        p0 = Sb - (Sqb * Sq) / Sq2
        w_count3 = 1.0 - (Sq * Sq) / Sq2
        w_count3 = w_count3.sum()

        pres0_mean = p0.sum() / w_count3
        p0 = pres0_mean

        J = Sq2 / (pres0_mean * Sq - Sqb)

        #p0 = (Sb * Sq2 - Sqb * Sq) / (Sq2 - Sq * Sq)
        #J = (Sq * Sq - Sq2) / (Sqb - Sb * Sq)
        #pres0_mean = p0.mean()

        J0 = J[J > 0.0].mean()
        Jstd = J[J > 0.0].std()
        J = np.where(J < 0.0, J0, J)
        J = np.where(J > J0 + 2 * Jstd, J0 + 2 * Jstd, J)

        perm = (tc.rand((well_count, well_count)) + 1.0) * (J0 + 1 * Jstd)
        perm = (perm + tc.transpose(perm, 0, 1)) * 0.5

        Jt = tc.from_numpy(J)
        perm_diag = tc.diag(perm, 0)
        perm += tc.diag(Jt - perm_diag)

        self.permeability.data = perm

        min_compr = tc.abs(Jt) * (self.dt * 2.0 / compr_mult)

        volume = (min_compr / tc.sum(min_compr)) * (vol_tot / vol_mult)
        self.volume.data = volume

        compressibility = (min_compr) * 0.01
        self.compressibility_o.data = compressibility
        self.compressibility_w.data = compressibility

        saturation0 = tc.from_numpy(sat_hist[0])
        saturation0 = tc.zeros(well_count)
        saturation0 = tc.logit((saturation0 + 0.01) / 1.02)
        self.saturation_init.data = saturation0

        pressure_well0 = tc.from_numpy(bhp_hist[0])
        production_well0 = tc.from_numpy(prod_hist[0])
        pressure_init = pressure_well0 + production_well0 / Jt
        self.pressure_init.data = pressure_init

        perm_aqu = tc.sqrt(tc.rand(well_count) + 1.0) * J0
        J_aqu = tc.tensor(J0)
        bhp_aqu = tc.tensor(pres0_mean)

        self.perm_aqu.data = perm_aqu
        self.J_aqu.data = J_aqu
        self.bhp_aqu.data = bhp_aqu

    def hidden_dim(self) -> int:
        return self.well_count

    def start(self):
        return tc.cat((self.pressure_init, self.saturation_init), 0)

    def calc_permeabilities_upwind(self, pressure: tc.Tensor, saturation: tc.Tensor, mob_prod_t: tc.Tensor):
        well_count = self.well_count

        pressure2 = pressure[:,None,:].expand((-1,  well_count, well_count ))
        mob_prod_t2 = mob_prod_t[:,None,:].expand((-1, well_count, well_count ))
        saturation2 = saturation[:,None,:].expand((-1, well_count, well_count ))

        upwind = (pressure2 > tc.transpose(pressure2, 1, 2))
        mob_prod_t2 = tc.where(upwind, mob_prod_t2, tc.transpose(mob_prod_t2, 1, 2))
        saturation2 = tc.where(upwind, saturation2, tc.transpose(saturation2, 1, 2))

        permeability_t = self.permeability * mob_prod_t2
        permeability_w = self.permeability * saturation2

        return permeability_t, permeability_w

    def calc_Q_in_base(self, permeability_t: tc.Tensor, permeability_w: tc.Tensor, pressure: tc.Tensor):
        S_t = permeability_t.sum(dim=2)
        S_w = permeability_w.sum(dim=2)        

        Q_in_t = tc.matmul(permeability_t, pressure[..., None])[..., 0] - S_t * pressure
        Q_in_w = tc.matmul(permeability_w, pressure[..., None])[..., 0] - S_w * pressure

        Q_in_t = Q_in_t - Q_in_t.mean(dim=1)[...,None]
        Q_in_w = Q_in_w - Q_in_w.mean(dim=1)[...,None]

        return Q_in_t, Q_in_w, S_t

    def step(self, input: tc.Tensor, param_input: tc.Tensor):
        well_count = self.well_count
        #pressure, x_saturation, production = input.split(well_count, dim=1)
        pressure, x_saturation = input.split(well_count, dim=1)
        production, wf = param_input.split(well_count, dim=1)

        saturation = tc.sigmoid(x_saturation)
        mob_prod_t = (saturation + (1.0 - saturation) * self.permeability_rel_o)

        J0 = tc.diag(self.permeability, 0)
        J_t = tc.where(production > 0, J0 * mob_prod_t, J0)
        J_w = tc.where(production > 0, J0 * saturation, J0)

        bhp_pred = pressure - production / J_t
        prod_w_pred = J_w * (pressure - bhp_pred)        

        permeability_t, permeability_w = self.calc_permeabilities_upwind(pressure, saturation, mob_prod_t)   

        Q_in_t, Q_in_w, S_t = self.calc_Q_in_base(permeability_t, permeability_w, pressure)     

        perm_aqu_abs = self.perm_aqu * self.perm_aqu
        prod_aqu = perm_aqu_abs * (self.bhp_aqu - pressure)

        Q_in_t = Q_in_t - production * wf + prod_aqu
        Q_in_w = Q_in_w - prod_w_pred * wf + prod_aqu

        compr = (self.compressibility_w * saturation + self.compressibility_o * (1.0 - saturation)) * self.compr_mult
        dpressure = self.dt * Q_in_t / (compr + self.dt * (S_t + perm_aqu_abs))
        pressure_next = pressure + dpressure

        dsaturation = (self.dt * Q_in_w - self.compressibility_w * self.compr_mult * saturation * dpressure) / (self.volume * self.vol_mult)
        x_saturation_next = x_saturation + dsaturation / ((saturation + 0.01) * (1.01 - saturation))

        return tc.cat((pressure_next, x_saturation_next), 1)

    def view(self, input: tc.Tensor, param_input: tc.Tensor):
        well_count = self.well_count
        #pressure, x_saturation, production = input.split(well_count, dim=1)
        pressure, x_saturation = input.split(well_count, dim=1)
        production, wf = param_input.split(well_count, dim=1)

        saturation = tc.sigmoid(x_saturation)
        mob_prod_t = (saturation + (1.0 - saturation) * self.permeability_rel_o)

        J0 = tc.diag(self.permeability, 0)
        J_t = tc.where(production > 0, J0 * mob_prod_t, J0)
        J_w = tc.where(production > 0, J0 * saturation, J0)

        bhp_pred = pressure - production / J_t
        prod_w_pred = J_w * (pressure - bhp_pred)

        return tc.cat((bhp_pred, prod_w_pred), 1)

