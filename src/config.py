import enums
from typing  import Dict, Any

class SchedulerConfig:
    def get_parameters(self) -> Dict[str, Any]:
      return vars(self)

    def __str__(self) -> str:
      return str(self.__class__)+":\n"+str(self.get_parameters())

    def __repr__(self) -> str:
      return str(self.__class__)+":\n"+str(self.get_parameters())

class StepLRConfig(SchedulerConfig):
  def __init__(self, step_size: int = 10, gamma = 0.1, last_epoch= -1, verbose=False):
    self.step_size = step_size
    self.gamma = gamma
    self.last_epoch = last_epoch
    self.verbose = verbose

class OptimizerConfig:
    def get_parameters(self) -> Dict[str, Any]:
      return vars(self)

    def __str__(self) -> str:
      return str(self.__class__)+":\n"+str(self.get_parameters())

    def __repr__(self) -> str:
      return str(self.__class__)+":\n"+str(self.get_parameters())

class AdamConfig(OptimizerConfig):
    def __init__(self, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, *, foreach=None, maximize=False, capturable=False):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.foreach = foreach
        self.maximize = maximize
        self.capturable = capturable


CFG= {
    "out_classes"      : 43,
    "optimizer"        : enums.Optimizer.Adam,
    "scheduler"        : None,
    "criterion"        : enums.Criterion.CrossEntropy,
    "epochs"           : 100,
    "early_stopping"   : 3,
    "optimizer_config" : AdamConfig(lr=0.05, amsgrad=True),
    "scheduler_config" : None,

    "train_batch_size" : 4,
    "test_batch_size"  : 4,

    "train_dir"        : "../res/preprocessed_data/train",
    "valid_dir"        : "../res/preprocessed_data/validation",
    "test_dir"         : "../res/preprocessed_data/test",

    "seed"             : 12,
    "model"            : enums.Model.ResNet50
}
