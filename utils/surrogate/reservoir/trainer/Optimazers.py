import torch as tc
import model.model_base as mb

import model.ModelInterface as mi

WEIGHT_DECAY = 1e-9
class ProxyOptimazer(mi.OptimazerInterface):
    def __init__(self, base: mb.ProxyModelBase):
        self.base = base
        self.optimazer = tc.optim.Adam(base.parameters(), lr=1e-4, betas=(0.5, 0.9))
        # self.optimazer = tc.optim.Adam(base.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay= WEIGHT_DECAY)

    def pre_step(self):
        #self.base.zero_grad()
        pass

    def step(self):
        #self.optimazer.step()
        pass

    def pre_epoch(self):
        self.base.zero_grad()

    def epoch(self):
        self.optimazer.step()

        perm = (self.base.permeability + tc.transpose(self.base.permeability, 0, 1)) * 0.5
        self.base.permeability.data = perm        

class ProxyOptimazerNet(mi.OptimazerInterface):
    def __init__(self, base: mb.ProxyModelNet, learning_rate_base: tc.float32, learning_rate_net: tc.float32):
        self.base = base
        self.optimazer = tc.optim.Adam(base.simple_params(), lr=learning_rate_base, betas=(0.9, 0.999))
        self.optimazerNet = tc.optim.Adam(base.net_params(), lr=learning_rate_net, betas=(0.9, 0.999))
        # self.optimazer = tc.optim.Adam(base.parameters(), lr=3e-4, betas=(0.5, 0.9), weight_decay= WEIGHT_DECAY)

    def pre_step(self):
        pass

    def step(self):
        pass

    def pre_epoch(self):
        self.base.saturation_init.data = tc.where(self.base.saturation_init < 1e-6, 1e-6, self.base.saturation_init)
        self.base.zero_grad()

    def epoch(self):
        if self.base.useNet == False:
            self.base.saturation_init.grad *= self.base.saturation_init

            #self.base.J.grad *= self.base.J
            #self.base.volume_inv.grad *= self.base.volume_inv
            #self.base.perm_aqu.grad *= self.base.perm_aqu       

            perm_grad = self.base.permeability.grad
            if not perm_grad is None:
                perm_grad = (perm_grad + tc.transpose(perm_grad, 0, 1))# * 0.5
                self.base.permeability.grad = perm_grad

        if self.base.useNet:
            weight_decay = 0.001

            for p in self.base.flowNet.parameters():
                if p.grad is not None:
                    p.grad += p * weight_decay

            self.optimazerNet.step()
        else:
            self.optimazer.step()