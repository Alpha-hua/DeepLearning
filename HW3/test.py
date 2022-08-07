from myModel import myModel
from tqdm import tqdm
from PIL import Image
from torchvision.datasets import DatasetFolder
from utils import test_tfm
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

if __name__ =="__main__":
    test_root = 'data/food-11/testing'
    bz = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "model/semi—acctensor(0.4534)"



    test_set = DatasetFolder(test_root, loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
    test_loader = DataLoader(test_set, batch_size=bz, shuffle=False)


    model = myModel()
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt)

    model.eval()
    predictions = []
    for batch in tqdm(test_loader):
        imgs, labels = batch

        with torch.no_grad():
            logits = model(imgs.to(device))

        predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    with open("predict.csv", "w") as f:

        f.write("Id,Category\n")

        for i, pred in enumerate(predictions):
            f.write(f"{i},{pred}\n")