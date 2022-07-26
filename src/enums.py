import enum

class Mode(enum.Enum):
    Train = 0
    Test  = 1

class Criterion(enum.Enum):
    CrossEntropy = 0

class Optimizer(enum.Enum):
    Adam = 0
    SGD  = 1

class Scheduler(enum.Enum):
    StepLR = 0

class Model(enum.Enum):
    ResNet18 = 0
    ResNet50 = 1
    OriginalCNN = 2

if __name__=="__main__":
    print(Mode.Train)

