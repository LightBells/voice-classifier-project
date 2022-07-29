from torchvision.transforms import Compose, ToTensor, Normalize
import enums


def getTransforms(*, mode=enums.Mode.Train):
    if mode == enums.Mode.Train:
        return Compose([ToTensor(), Normalize((0.5), (0.5))])
    elif mode == enums.Mode.Test:
        return Compose(
            [
                ToTensor(),
                Normalize((0.5), (0.5)),
            ]
        )
    else:
        raise NotImplementedError("Selected Transforms mode have not defined.")


if __name__ == "__main__":
    print("For Train: ")
    print(getTransforms(mode=enums.Mode.Train))
    print("For Test: ")
    print(getTransforms(mode=enums.Mode.Test))
