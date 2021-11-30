import cv2
from torchvision import transforms
from PIL import Image
'''
通过transforms.ToTensor去看两个问题
（1）transforms如何使用
（2）为什么需要tensor数据类型
'''
img_path = "/home/lr/torch/dataset/hymenoptera_data/train/ants/5650366_e22b7e1065.jpg"
# Image是python内置的一个看图片的库，使用Image产生的数据就是PIL Image
# 如果使用OpenCV，则产生的数据类型是ndarry，transforms.ToTensor()也可以转换
img = Image.open(img_path)
print(img)

# 实例化一个对象，这个实例名字叫tensor_trans，tensor_trans是一个实例，而不是一个类
tensor_trans = transforms.ToTensor()
# tensor_trans这个实例在调用__repr__方法
print(tensor_trans)
# tensor_trans这个实例在调用__call__方法，调用__call__方法就是直接在实例后面括号中加上
# __call__方法所需要的参数，而这个参数就是img，通过__call__方法将其转变成tensor
img_tensor = tensor_trans(img)
print(img_tensor)