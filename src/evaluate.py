from train import test
from models import ResNet18, ResNet50, OriginalCNN, EfficientNetB0
from dataset import VoiceDataset
from transforms import getTransforms
from enums import Mode
from config import CFG

import argparse
import torch
import numpy as np

import pprint

parser = argparse.ArgumentParser(description="model evaluation script.")
parser.add_argument(
    "model_name",
    type=str,
    help="Specify a model type, \
            [ResNet18, ResNet50, OriginalCNN, EfficientNetB0]",
)
parser.add_argument("weight_path", type=str, help="Specify model weight path.")
parser.add_argument("output_classes", type=int)
args = parser.parse_args()


if __name__ == "__main__":
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    if args.model_name == "ResNet18":
        model = ResNet18(args.output_classes)
    elif args.model_name == "ResNet50":
        model = ResNet50(args.output_classes)
    elif args.model_name == "OriginalCNN":
        model = OriginalCNN(args.output_classes)
    elif args.model_name == "EfficientNetB0":
        model = EfficientNetB0(args.output_classes)
    else:
        raise NotImplementedError("The model type is not supported.")

    weight = torch.load(args.weight_path)
    model.load_state_dict(weight)
    model.to(device)

    test_ds = VoiceDataset(CFG["test_dir"],
                           transforms=getTransforms(mode=Mode.Test))

    metrics, result = test(model, test_ds, device)

    pprint.pprint(metrics)

    correct = np.sum(result[0] == result[1])
    all = len(result[0])
    print(f"Correct/ALL: {correct}/{all}")
