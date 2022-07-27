import argparse
from models import ResNet18, ResNet50, OriginalCNN, EfficientNetB0

parser = argparse.ArgumentParser(description="model evaluation script.")
parser.add_argument("model_name", type=str, help="Specify a model type, [ResNet18, ResNet50, OriginalCNN, EfficientNetB0]")
parser.add_argument("weight_path", type=str, help="Specify model weight path.")
parser.add_argument("output_classes", type=int)
args = parser.parse_args()

import torch
from train import test

if __name__=="__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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

    metrices = test(model, device)
    
    print(metrices)
