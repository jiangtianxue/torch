import torchvision
from torch.utils.data import DataLoader

# 准备的测试集
test_set = torchvision.datasets.CIFAR10(
    './cifar10_set',
    train = False,
    transform = torchvision.transforms.ToTensor()
)

test_loader = DataLoader(
    dataset = test_set,
    batch_size = 4,
    shuffle = True,
    num_workers = 0,
    drop_last = False
)

# 测试集中第一个数据的具体内容
img, target = test_set[0]
print(img.shape)
print(target)

for data in test_loader:
    print(type(data))
    print(len(data))
    print(data)
    imgs, targets = data
    print(type(imgs))
    print(imgs.shape)
    print(imgs)
    print(targets)

for i in range(epoch):
    for imgs, targets in test_loader:
        pred = model(imgs)