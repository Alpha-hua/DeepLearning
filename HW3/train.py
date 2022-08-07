import copy

from utils import train_tfm, test_tfm
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.datasets import DatasetFolder
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm
from myModel import  myModel

def get_pseudo_labels(dataset, model, threshold=0.8):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bz = 128
    newDataSet = copy.deepcopy(dataset)
    samples = []
    targets = []
    model.eval()
    softmax = nn.Softmax(dim=-1)
    unlabeled_loader = DataLoader(dataset,batch_size=bz,shuffle=True)
    for i,batch in enumerate(tqdm(unlabeled_loader)):
        img, _ = batch

        with torch.no_grad():
            preds = model(img.to(device))

        prediction, index = torch.max(softmax(preds),dim=-1)
        if len(img) == bz:
            for j in range(bz):
                newDataSet.targets[i*bz+j] = (int(index[j]) if prediction[j]>threshold else 0)
        else:
            for j in range(len(img)):
                newDataSet.targets[i * bz + j] = (int(index[j]) if prediction[j] > threshold else 0)


    for i in range(len(newDataSet)):
        if newDataSet.targets[i] != 0:
            samples.append(newDataSet.samples[i])
            targets.append(newDataSet.targets[i])
    newDataSet.samples,newDataSet.targets = samples,targets
    print("添加{}个数据".format(len(newDataSet)))
    model.train()
    return newDataSet


if __name__ == '__main__':
    softmax = nn.Softmax(dim=-1)
    bz = 128
    train_root = 'data/food-11/training/labeled'
    unlabeled_root = "data/food-11/training/unlabeled"
    val_root = 'data/food-11/validation'
    model_path = "model/semi—acctensor(0.4437)"
    # model_path = ""
    train_set = DatasetFolder(train_root, loader=lambda x: Image.open(x), extensions="jpg",
                              transform=train_tfm)
    valid_set = DatasetFolder(val_root, loader=lambda x: Image.open(x), extensions="jpg",
                              transform=test_tfm)
    unlabeled_set = DatasetFolder(unlabeled_root, loader=lambda x: Image.open(x), extensions="jpg",
                                  transform=train_tfm)

    train_loader = DataLoader(train_set, batch_size=bz, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=bz, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = myModel().to(device)
    model.device = device
    if model_path != "":
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)

    n_epochs = 60

    do_semi = True

    best_acc = 0

    for epoch in range(n_epochs):
        if do_semi:
            pseudo_set = get_pseudo_labels(unlabeled_set, model)
            concat_dataset = ConcatDataset([train_set, pseudo_set])
            train_loader = DataLoader(concat_dataset, batch_size=bz, shuffle=True)

        model.train()

        train_loss = []
        train_accs = []

        for batch in tqdm(train_loader):
            imgs, labels = batch

            preds = model(imgs.to(device))

            loss = criterion(preds, labels.to(device))

            optimizer.zero_grad()

            loss.backward()

            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            optimizer.step()

            acc = (preds.argmax(dim=-1) == labels.to(device)).float().mean()

            train_loss.append(loss.item())
            train_accs.append(acc)

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")




        model.eval()
        valid_loss = []
        valid_accs = []

        for batch in tqdm(valid_loader):
            imgs, labels = batch

            with torch.no_grad():
                logits = model(imgs.to(device))

            loss = criterion(logits, labels.to(device))

            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            valid_loss.append(loss.item())
            valid_accs.append(acc)

        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
        if valid_acc>best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), 'model/semi—acc' + str(valid_acc))
            print('saving model with accuracy {:.3f}'.format(float(valid_acc)))

