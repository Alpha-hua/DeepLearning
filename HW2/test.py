import torch
from model import myModel
import pandas as pd
import numpy as np

if __name__ == '__main__':
    test_path = "data/test_11.npy"
    model_path = "model/acc0.716438499717238"

    data = np.load(test_path)
    ckpt = torch.load(model_path)
    model = myModel()
    model.load_state_dict(ckpt)
    model.eval()
    x = torch.from_numpy(np.float32(data)).to("cpu")
    with torch.no_grad():
        pred = model(x)
    _, test_pred = torch.max(pred, 1)

    last_data = torch.cat((torch.arange(start=0, end=test_pred.shape[0], step=1).unsqueeze(1),test_pred.unsqueeze(1)),1)

    a = pd.DataFrame(last_data.numpy())
    a.to_csv("predict.csv", index=False, header=['Id','Class'])
    print()