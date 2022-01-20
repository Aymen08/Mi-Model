import torch

class Ema:
    r"""
    Implements ema algorithm.
    """

    def __init__(self, model, decay=0.98):
        super(Ema, self).__init__()
        self.decay = decay
        self.model = model
        self.old_params = {}
        for name, params in self.model.named_parameters():
            self.old_params[name] = params.clone()
        
    @torch.no_grad()
    def update(self):
        """
        Performs a single optimization step.
        """
        for name, params in self.model.named_parameters():
            self.old_params[name] = (1.0 - self.decay) * params.data + self.decay * self.old_params[name]
