import torchvision

# 对数据集进行转换了，这里只进行了ToTensor的转换
# 还可以进行很多其他转换，这里没有体现
dataset_transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()]
)

# 通过CIFAR10这个类实例化train_set和test_set两个实例
# 并且对每一个数据（即每一张图片）都进行dataset_transform的转换
# 也就是将数据类型转换为tensor
train_set = torchvision.datasets.CIFAR10(
    root = "./cifar10_set",
    train = True,
    transform = dataset_transform,
    download = True
)

test_set = torchvision.datasets.CIFAR10(
    root = "./cifar10_set",
    train = False,
    transform = dataset_transform,
    download = True
)
# 注意，train_set和test_set都是类实例化产生的实例
print(train_set[0]); print(type(train_set[0]))
print(train_set.classes)
img, target = train_set[0]

# 因为进行了数据转换，所以此时的img，也就是数据主体
# 的数据类型已经是tensor了
print(img); print(type(img))
print(target); print(type(target))
print(train_set.classes[target])