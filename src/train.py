from models import ResNet18, OriginalCNN, ResNet50, EfficientNetB0
from transforms import getTransforms
from dataset import VoiceDataset
from config import CFG
from utils.seed import seed_torch
import enums

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
)


def train_fn(
    epoch, model, data_loader, optimizer, scheduler, criterion, device
):
    total_loss = 0.0
    running_loss = 0.0
    total = 0
    correct = 0
    model.train()
    for idx, (x, d) in enumerate(data_loader):
        x, d = x.to(device), d.to(device)
        y = model(x)

        loss = criterion(y, d)

        preds = torch.softmax(y, dim=1)
        total += x.shape[0]
        correct += torch.sum(preds.argmax(dim=1) == d)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        running_loss += loss.item()

        if idx % 16 == 15:
            print(
                f"Train:: Epoch {epoch}, Batch {idx}/{len(data_loader)}, \
                Running Loss Avg: {running_loss/16} Accuracy: {correct/total}"
            )
            running_loss = 0.0

    if scheduler is not None:
        scheduler.step()

    print(
        f"Train:: Epoch {epoch}, Batch {idx}/{len(data_loader)}, \
        Total Loss Avg: {total_loss/len(data_loader)} \
        Total Accuracy: {correct/total}"
    )

    writer.add_scalar("Loss/train", total_loss / len(data_loader), epoch)
    writer.add_scalar("Accuracy/train", correct / total, epoch)
    return model, (total_loss / len(data_loader)), correct / total


def valid_fn(epoch, model, data_loader, criterion, device):
    total_loss = 0.0
    running_loss = 0.0
    total = 0
    correct = 0
    model.eval()
    for idx, (x, d) in enumerate(data_loader):
        x, d = x.to(device), d.to(device)
        with torch.no_grad():
            y = model(x)
            loss = criterion(y, d)

        preds = torch.softmax(y, dim=1)
        total += x.shape[0]
        correct += torch.sum(preds.argmax(dim=1) == d)

        total_loss += loss.item()
        running_loss += loss.item()

        if idx % 16 == 15:
            print(
                f"Validation:: Epoch {epoch}, Batch {idx}/{len(data_loader)}, \
                        Running Loss Avg: {running_loss/16} \
                        Accuracy: {correct/total}"
            )
            running_loss = 0.0

    print(
        f"Validation:: Epoch {epoch}, Batch {idx}/{len(data_loader)}, \
                Total Loss Avg: {total_loss/len(data_loader)} \
                Total Accuracy: {correct/total}"
    )

    writer.add_scalar("Loss/valid", total_loss / len(data_loader), epoch)
    writer.add_scalar("Accuracy/vaild", correct / total, epoch)
    return model, (total_loss / len(data_loader)), correct / total


def fit(model, epochs, optimizer, scheduler, criterion, device):
    train_ds = VoiceDataset(
        CFG["train_dir"], transforms=getTransforms(mode=enums.Mode.Train)
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=CFG["train_batch_size"],
        shuffle=True,
        num_workers=2,
    )

    valid_ds = VoiceDataset(
        CFG["valid_dir"], transforms=getTransforms(mode=enums.Mode.Test)
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=CFG["test_batch_size"],
        shuffle=False,
        num_workers=2,
    )

    best_loss = 10000.0
    best_weight = None
    early_stopping = CFG["early_stopping"]
    for epoch in range(1, epochs + 1):
        model, train_loss, accuracy = train_fn(
            epoch, model, train_loader, optimizer, scheduler, criterion, device
        )

        model, valid_loss, valid_accuracy = valid_fn(
            epoch, model, valid_loader, criterion, device
        )
        if valid_loss < best_loss:
            torch.save(model.state_dict(), "best_model.pth")
            best_loss = valid_loss
            best_weight = model.state_dict()
            print(f"Best Loss is Upgated to {valid_loss}")
            early_stopping = CFG["early_stopping"]
        else:
            early_stopping -= 1

        if early_stopping <= 0:
            break

    model.load_state_dict(best_weight)
    return model


def test(model, device):
    test_ds = VoiceDataset(
        CFG["test_dir"], transforms=getTransforms(mode=enums.Mode.Test)
    )
    data_loader = DataLoader(
        test_ds, batch_size=CFG["test_batch_size"], shuffle=False
    )

    model.eval()
    predictions = []
    actual = []
    features = []
    for idx, (x, d) in enumerate(data_loader):
        x, d = x.to(device), d.to(device)
        with torch.no_grad():
            y = model(x)

        preds = torch.softmax(y, dim=1)
        predictions.append(preds.argmax(dim=1).to("cpu").detach().numpy())
        features.append(preds.to("cpu").detach().numpy())
        actual.append(d.to("cpu").numpy())

    actual = np.concatenate(actual)
    predictions = np.concatenate(predictions)
    features = np.concatenate(features)

    print(actual)
    print(predictions)

    metrics = {
        "accuracy": accuracy_score(actual, predictions),
        "micro/precision": precision_score(
            actual, predictions, average="micro"
        ),
        "micro/recall": recall_score(actual, predictions, average="micro"),
        "micro/f1_score": f1_score(actual, predictions, average="micro"),
        "macro/precision": precision_score(
            actual, predictions, average="macro"
        ),
        "macro/recall": recall_score(actual, predictions, average="macro"),
        "macro/f1_score": f1_score(actual, predictions, average="macro"),
    }

    return metrics, (actual, predictions, features)


def main():
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    if CFG["model"] == enums.Model.ResNet18:
        model = ResNet18(CFG["out_classes"])
    elif CFG["model"] == enums.Model.ResNet50:
        model = ResNet50(CFG["out_classes"])
    elif CFG["model"] == enums.Model.EfficientNetB0:
        model = EfficientNetB0(CFG["out_classes"])
    elif CFG["model"] == enums.Model.OriginalCNN:
        model = OriginalCNN(CFG["out_classes"])
    else:
        raise NotImplementedError("The Model is not Impemented")
    model.to(device)

    # Optimizer Initialize
    if CFG["optimizer"] == enums.Optimizer.Adam:
        optimizer = optim.Adam(
            model.parameters(), **CFG["optimizer_config"].get_parameters()
        )
    else:
        raise NotImplementedError("The optimizer is not Impemented")

    from schedulers import LinearCyclicalLR
    # Scheduler Initialize
    if CFG["scheduler"] is None:
        scheduler = None
    elif CFG["scheduler"] == enums.Scheduler.StepLR:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, **CFG["scheduler_config"].get_parameters()
        )
    elif CFG["scheduler"] == enums.Scheduler.LinearCyclicalLR:
        scheduler = LinearCyclicalLR(
            optimizer, **CFG["scheduler_config"].get_parameters()
        )
    else:
        raise NotImplementedError("The scheduler is not Impemented")

    # Criterion Setting
    if CFG["criterion"] == enums.Criterion.CrossEntropy:
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError("The scheduler is not Impemented")

    model = fit(model, CFG["epochs"], optimizer, scheduler, criterion, device)

    metrics, results = test(model, device)
    print(metrics)


if __name__ == "__main__":
    seed_torch(CFG["seed"])
    writer = SummaryWriter()

    # Record Settings and Hyper parameters
    for key in CFG:
        writer.add_text(key, str(CFG[key]), 0)

    main()
