from models import Net
from transforms import getTransforms
from dataset import VoiceDataset
from config import CFG
import enums

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

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


        if scheduler is not None:
            scheduler.step()

        if idx % 16 == 15:
            print(f"Train:: Epoch {epoch}, Batch {idx}/{len(data_loader)}, Running Loss Avg: {running_loss/16} Accuracy: {correct/total}")
            running_loss = 0.0

    print(f"Train:: Epoch {epoch}, Batch {idx}/{len(data_loader)}, Total Loss Avg: {total_loss/len(data_loader)} Total Accuracy: {correct/total}")
    return model, (total_loss/len(data_loader)), correct/total

def valid_fn(
        epoch, model, data_loader, criterion, device
):
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
            print(f"Validation:: Epoch {epoch}, Batch {idx}/{len(data_loader)}, Running Loss Avg: {running_loss/16} Accuracy: {correct/total}")
            running_loss = 0.0

    print(f"Validation:: Epoch {epoch}, Batch {idx}/{len(data_loader)}, Total Loss Avg: {total_loss/len(data_loader)} Total Accuracy: {correct/total}")

    return model, (total_loss/len(data_loader)), correct/total


def fit(model, epochs, optimizer, scheduler, criterion, device):
    train_ds     = VoiceDataset(CFG["train_dir"], transforms=getTransforms(mode=enums.Mode.Train))
    train_loader = DataLoader(train_ds, batch_size=8, 
            shuffle=True, num_workers=2)

    valid_ds     = VoiceDataset(CFG["valid_dir"], transforms=getTransforms(mode=enums.Mode.Test))
    valid_loader = DataLoader(valid_ds, batch_size=8, 
            shuffle=False, num_workers=2)


    best_loss = 10000.0
    early_stopping = CFG["early_stopping"]
    for epoch in range(1, epochs+1):
        model, train_loss, accuracy = train_fn(epoch, model, 
                train_loader,optimizer, scheduler, criterion, device)

        model, valid_loss, valid_accuracy = valid_fn(epoch, model,
                valid_loader, criterion, device)
        if valid_loss < best_loss:
            torch.save(model, "best_model.pth")
            best_loss = valid_loss
            print(f"Best Loss is Upgated to {valid_loss}")
            early_stopping = CFG["early_stopping"]
        else:
            early_stopping -= 1

        if early_stopping <= 0:
            break
        

def test(model):
    model.eval()


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = Net(CFG["out_classes"])
    model.to(device)
    
    # Optimizer Initialize
    if CFG["optimizer"] == enums.Optimizer.Adam:
        optimizer = optim.Adam(model.parameters(), **CFG["optimizer_config"].get_parameters())
    else:
        raise NotImplementedError("The optimizer is not Impemented")

    # Scheduler Initialize
    if CFG["scheduler"] is None:
        scheduler = None
    elif CFG["scheduler"] == enums.Scheduler.StepLR:
        scheduler = optim.lr_scheduler.StepLR(optimizer, **CFG["scheduler_config"].get_parameters())
    else:
        raise NotImplementedError("The scheduler is not Impemented")
    
    # Criterion Setting
    if CFG["criterion"] == enums.Criterion.CrossEntropy:
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError("The scheduler is not Impemented")

    model = fit(model, CFG["epochs"], optimizer, scheduler, criterion, device)

    metrices = test(model)
    print(metrices)

if __name__=="__main__":
    main()
