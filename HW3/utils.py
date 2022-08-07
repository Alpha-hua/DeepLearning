import torchvision.transforms as transforms



train_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),# 左右反转
    transforms.ColorJitter(brightness=.5, hue=.3), #ColorJitter变换随机更改图像的亮度、饱和度和其他属性。
    transforms.ToTensor(),
])

test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])