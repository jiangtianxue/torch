import os
from torch.utils.data import Dataset
from PIL import Image

class MyData(Dataset):
    # 类实例化的初始化函数，将传入的root_dir,label_dir参数传入实例
    # 以便于后面的__getitem__方法使用
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    # 类函数的重写，获得每一个数据本身及其label
    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        # 获得每一个数据的路径
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        # 读取每一个数据
        img = Image.open(img_item_path)
        # 读取每一个数据的label
        label = self.label_dir
        # 返回数据及其label
        return img, label
    
    # 类函数的重写，获得数据集的长度
    def __len__(self):
        return len(self.img_path)

root_dir = "/home/lr/torch/dataset/hymenoptera_data/train"
label_dir1 = "ants"
label_dir2 = "bees"
ants_dataset = MyData(root_dir, label_dir1)
bees_dataset = MyData(root_dir, label_dir2)

print(len(ants_dataset))
print(ants_dataset.__len__())

print(len(bees_dataset))

train_dataset = ants_dataset + bees_dataset

# 这就是__len__方法的调用方式
print(len(train_dataset))

# 传入的参数1就是idx，所以这就是__getitem__方法的调用方式
img, label = ants_dataset[1]
img.show()
