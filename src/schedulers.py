from torch.optim.lr_scheduler import _LRScheduler


class LinearCyclicalLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super(LinearCyclicalLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        last_step = self.last_epoch % self.T_max
        point = self.T_max // 2
        return [
            -(base_lr - self.eta_min) * last_step / (point - 0.5) + base_lr
            if last_step < point
            else (base_lr - self.eta_min) * last_step / (point - 0.5)
            + self.eta_min
            - base_lr
            for base_lr in self.base_lrs
        ]
