from myDataSet import myDataSet
import numpy as np
from torch.utils.data import DataLoader
from model import myModel
import torch
import torch.nn as nn
from tqdm import tqdm

if __name__ == '__main__':
    data = np.load('data/train_11.npy')
    label = np.load('data/train_label_11.npy')
    model_path = 'model/acc0.722'
    bz = 64
    # data, label = data[:int(len(data)*0.1)], label[:int(len(label)*0.1)]
    train_num = int(len(data)*0.9)
    train_x, train_y, val_x, val_y = data[:train_num], label[:train_num], data[train_num:], label[train_num:]

    train_dataSet = myDataSet(train_x, train_y)
    train_dataLoader = DataLoader(dataset=train_dataSet, batch_size=bz, shuffle=True)
    val_dataSet = myDataSet(val_x, val_y)
    val_dataLoader = DataLoader(dataset=val_dataSet, batch_size=bz, shuffle=False)

    model = myModel()
    #ckpt = torch.load(model_path)
    #model.load_state_dict(ckpt)
    num_epoch = 10
    lr = 0.0001
    Loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best = 0
    for epoch in range(num_epoch):
        model.train()
        total_loss = 0
        train_acc = 0
        with tqdm(total=len(train_dataLoader), desc=f'training epoch:{epoch+1}', postfix=dict, mininterval=0.3) as pbar:

            for iter, data in enumerate(train_dataLoader):
                optimizer.zero_grad()
                x = data[0]
                y = data[1]
                pred = model(x)
                loss = Loss(pred,y)
                _, train_pred = torch.max(pred, 1) # 按第一维取最大值，保留索引
                loss.backward()
                optimizer.step()

                train_acc += (train_pred.cpu() == y.cpu()).sum().item()
                total_loss += loss.item()
                pbar.set_postfix(**{'loss': total_loss/(iter+1), 'acc': train_acc/(bz*(iter+1))})
                pbar.update(1)


        val_loss = 0
        val_acc = 0

        with tqdm(total=len(val_dataLoader), desc=f'val epoch:{epoch + 1}', postfix=dict,mininterval=0.3) as pbar:
            with torch.no_grad():
                for iter, data in enumerate(val_dataLoader):
                    x,y=data[0],data[1]
                    pred = model(x)
                    loss = Loss(pred, y)
                    _, val_pred = torch.max(pred, 1)  # 按第一维取最大值，保留索引
                    val_acc += (val_pred.cpu() == y.cpu()).sum().item()
                    val_loss += loss.item()
                    pbar.set_postfix(**{'loss': val_loss / (iter + 1), 'acc': val_acc /(bz*(iter+1))})
                    pbar.update(1)

        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), 'model/acc'+str(best / (train_num/9)))
            print('saving model with accuracy {:.3f}'.format(best / (train_num/9)))


