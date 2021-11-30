import cv2
from PIL import Image
from torchvision import transforms

img_path = "/home/lr/torch/dataset/hymenoptera_data/train/ants/5650366_e22b7e1065.jpg"
# Image是python内置的一个看图片的库，使用Image产生的数据就是PIL Image
# 如果使用OpenCV，则产生的数据类型是ndarry
# transforms.ToTensor()可以将这两种格式转换为tensor数据类型
img = Image.open(img_path)
print("使用PIL打开图片的类型是："); print(img)
img_cv2 = cv2.imread(img_path)
print("使用opencv打开图片的类型是："); print(type(img_cv2))

# ToTensor()类
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
# print(img_tensor)

# Normalize()类
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
# 下面一行也是调用__call__方法
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])


# Resize()类，输入为PIL数据类型
print(img.size)
trans_resize = transforms.Resize((512, 512))
# 下面一行还是调用了__call__方法
img_resize = trans_resize(img)
# 下面两个是不一样的东西
print(img_resize)
print(trans_resize)

# PIL -> resize -> PIL -> totensor -> tensor数据类型
img_resize = trans_totensor(img_resize) 


# Compose类，实现组合转变方式
trans_resize_2 = transforms.Resize(512)
# 先进行trans_resize_2，再进行trans_totensor
# 所以要确保第一个的输出类型和第二个的输入类型是匹配的
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
print(img_resize_2[0][0][0])


# RandomCrop类，参数是一个序列（h, w）,会裁剪成（h, w）
# 的图片；或者是一个size，会裁剪成（size, size）的图片
trans_random = transforms.RandomCrop(128)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    # 数据类型由 PIL -> PIL -> tensor
    img_crop = trans_compose_2(img)
    print(img_crop[0][0][0])
