import torch as tc
import torch.nn as nn
import numpy as np
import model.ModelInterface as mi

class MultiTrainer:
    model: mi.SModelInterface
    model_test: mi.SModelInterface
    disps_view0: tc.Tensor
    disps_hidden0: tc.Tensor
    
    def __init__(self, model: mi.SModelInterface, model_test: mi.SModelInterface, optimizer, device= tc.device('cpu')):
        self.model = model
        self.model_test = model_test
        self.optimizer = optimizer
        self.device = device

        view_dim = model.view_dim()
        self.disps_view0 = tc.ones(view_dim)

        hidden_dim = model.hidden_dim()
        self.disps_hidden0 = tc.ones(hidden_dim)

    def set_disps_view(self, disps_view: np.ndarray):        
        disps_view = tc.from_numpy(disps_view).type(tc.float32)
        self.disps_view0 = disps_view.to(device= self.device)

        disps_view = tc.diag(disps_view * disps_view)

    def set_disps_hidden(self, disps_hidden: np.ndarray):
        disps_hidden = tc.from_numpy(disps_hidden).type(tc.float32)
        self.disps_hidden0 = disps_hidden.to(device= self.device)

        disps_hidden = tc.diag(disps_hidden * disps_hidden)

    def _history_step(self, time_step: int, hidden_prev: tc.Tensor):
        if time_step > 0:
            hidden_cur = self.model_test.step(time_step - 1, hidden_prev)
        else:
            hidden_cur = self.model_test.start()

        self.model_test.save_hidden(time_step, hidden_cur)

        view = self.model_test.view(time_step, hidden_cur)
        self.model_test.save_view(time_step, view)

        return hidden_cur

    @tc.inference_mode()
    def make_history(self):
        time_steps = self.model_test.time_steps()

        hidden = None
        for time_step in range(time_steps):
            hidden = self._history_step(time_step, hidden)

    def step_ml(self, time_step: int, hidden_prev):
        if time_step > 0:
            hidden_cur = self.model.step(time_step - 1, hidden_prev)
        else:
            hidden_cur = self.model.start()

        cur_view = self.model.view(time_step, hidden_cur)
        view_true = self.model.true_view(time_step)
        view_weights = self.model.veiw_weights(time_step, hidden_cur)

        delta_v = (cur_view - view_true) / self.disps_view0
        delta_v = tc.where(tc.isnan(delta_v), 0.0, delta_v)
        loss_v = (delta_v * delta_v * view_weights).mean()

        loss_v_v = loss_v.detach().cpu().numpy()

        if np.isnan(loss_v_v) or tc.isnan(hidden_cur.mean()) or loss_v_v > 10000.0:
            self.step_ml(time_step, hidden_prev)
            print(time_step)

        return hidden_cur, loss_v, loss_v_v

    def epoch_ml(self, prob = 1.0):
        time_steps = self.model.time_steps()
        loss_full = 0.0

        hidden_prev = None
        loss_s = tc.zeros(1).to(device= self.device)

        self.optimizer.pre_epoch()

        for time_step in range(time_steps):
            self.optimizer.pre_step()

            hidden_cur, loss, loss_val = self.step_ml(time_step, hidden_prev)

            if np.isnan(loss_val):
                return loss_val

            #if time_step == 0:
            #    loss *= time_steps
            #    loss_val *= time_steps
            #
            #    #loss_s += loss
            #    loss_s = loss_s * 0.98 + loss
            #else:
            #    if np.random.uniform(0.0, 1.0) < prob:
            #        loss_s = loss_s * 0.98 + loss

            #if np.random.uniform(0.0, 1.0) < prob:
            loss_s = loss_s + loss
            #loss_s = loss_s * 0.98 + loss
            
            loss_full += loss_val
            #loss_full = loss_full * 0.98 + loss_val

            hidden_prev = hidden_cur

            self.optimizer.step()

        loss_s.backward()
        self.optimizer.epoch()

        loss_full /= time_steps
        #loss_full *= 0.02
        return loss_full  

    @tc.inference_mode()
    def step_test(self, time_step: int, hidden_prev):
        if time_step > 0:
            hidden_cur = self.model_test.step(time_step - 1, hidden_prev)
        else:
            hidden_cur = self.model_test.start()

        self.model_test.save_hidden(time_step, hidden_cur)
        self.model_test.save_hidden_cor(time_step, hidden_cur)

        cur_view = self.model_test.view(time_step, hidden_cur)
        view_true = self.model_test.true_view(time_step)
        view_weights = self.model_test.veiw_weights(time_step, hidden_cur)

        delta_v = (cur_view - view_true) / self.disps_view0
        #loss_v = tc.nanmean(delta_v * delta_v * view_weights)
        delta_v = tc.where(tc.isnan(view_true), 0.0, delta_v)
        loss_v = (delta_v * delta_v * view_weights).mean()

        loss_v_v = loss_v.detach().cpu().numpy()

        if np.isnan(loss_v_v) or tc.isnan(hidden_cur.mean()) or loss_v_v > 10000.0:
            self.step_test(time_step, hidden_prev)
            print(time_step)

        return hidden_cur, loss_v_v

    @tc.inference_mode()
    def epoch_test(self):
        time_steps = self.model_test.time_steps()

        hidden_prev = None
        loss_s = 0.0

        for time_step in range(time_steps):
            hidden_cur, loss = self.step_test(time_step, hidden_prev)

            if np.isnan(loss):
                return loss

            #if time_step == 0:
            #    loss *= time_steps

            loss_s += loss
            #loss_s = loss_s * 0.98 + loss
            hidden_prev = hidden_cur

        loss_s /= time_steps
        #loss_s *= 0.02
        return loss_s
