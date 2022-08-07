import torch
from MyModel import Model
import pandas as pd
import numpy as np

if __name__ == '__main__':
    test_path = "data/covid.test.csv"

    dataset = pd.read_csv(test_path)
    dataset = np.array(dataset.values.tolist())
    ckpt = torch.load("model/mymodel")
    model = Model(94)
    model.load_state_dict(ckpt)
    model.eval()
    x = torch.from_numpy(np.float32(dataset)).to("cpu")
    with torch.no_grad():
        x.unsqueeze(0)
        pred = model(x)
    pred = pred.numpy()
    a = pd.DataFrame(pred)
    a.to_csv("predict.csv", header=['tested_positive'])
    print()