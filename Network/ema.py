import torch
import torch.nn as nn
from copy import deepcopy


class EMA(nn.Module):
    def __init__(self, model, decay=0.999, device=None):
        super(EMA, self).__init__()
        self.module = deepcopy(model) # make a copy of the model for accumulating moving average of weights
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)
        else:
            self.module.to(device=model.device)

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
