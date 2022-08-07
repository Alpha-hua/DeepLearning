from MyDataset import MyDataset
from MyModel import Model
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np

if __name__ == '__main__':
    train_path = "data/covid.train.csv"
    train_dataset = MyDataset(train_path)
    train_dataset = DataLoader(dataset=train_dataset, batch_size=270, shuffle=True)


    '''val_path = "data/covid.val.csv" 
    val_dataset = MyDataset(val_path)
    val_dataset = DataLoader(dataset=val_dataset, batch_size=270, shuffle=True)'''


    model = Model(94).to('cpu')
    MSE = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.000005)
    for epoch in range(10000):
        model.train()
        total_loss = 0
        for x, y in train_dataset:
            x, y = x.to("cpu"), y.to("cpu")
            optimizer.zero_grad()
            pred = model(x)
            loss = MSE(pred,y)
            total_loss+=loss
            loss.backward()
            optimizer.step()
        print("epoch:{},train loss:{}".format(epoch,total_loss/9))

        '''model.eval()
        for x,y in val_dataset:
            with torch.no_grad():
                pred = model(x)
                loss = MSE(pred,y)
            print("val loss:{}".format(loss))'''
    torch.save(model.state_dict(), "model/mymodel")
