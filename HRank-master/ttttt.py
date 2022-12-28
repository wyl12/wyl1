
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

plt.ion()



data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(84),
        transforms.CenterCrop(84),
        # 转换成tensor向量
        transforms.ToTensor(),
        # 对图像进行归一化操作
        # [0.485, 0.456, 0.406]，RGB通道的均值与标准差
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(84),
        transforms.CenterCrop(84),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
def Load_Image_Information(path):
    # 图像存储路径
    image_Root_Dir= r'/dataset/data/data/'
    # 获取图像的路径
    iamge_Dir = os.path.join(image_Root_Dir, path)
    # 以RGB格式打开图像
    # Pytorch DataLoader就是使用PIL所读取的图像格式

    img = Image.open(iamge_Dir)
    img = img.resize(size=(128, 128)).convert('RGB')
    return img


class my_Data_Set(nn.Module):
    def __init__(self, txt, transform=None, target_transform=None, loader=None):
        super(my_Data_Set, self).__init__()
        # 打开存储图像名与标签的txt文件
        fp = open(txt, 'r')
        images = []
        labels = []
        # 将图像名和图像标签对应存储起来
        for line in fp:
            line.strip('\n')
            line.rstrip()
            information = line.split()
            images.append(information[0])
            labels.append(int(information[1]))
        self.images = images
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    # 重写这个函数用来进行图像数据的读取
    def __getitem__(self, item):
        # 获取图像名和标签
        imageName = self.images[item]
        label = self.labels[item]
        # 读入图像信息
        image = self.loader(imageName)
        # 处理图像数据
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    # 重写这个函数，来看数据集中含有多少数据
    def __len__(self):
        return len(self.images)

#
# # 生成Pytorch所需的DataLoader数据输入格式
# train_dataset = my_Data_Set('train.txt', transform=data_transforms['train'], loader=Load_Image_Information)
# test_dataset = my_Data_Set('val.txt', transform=data_transforms['val'], loader=Load_Image_Information)
# train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

'''
# 验证是否生成DataLoader格式数据
for data in train_loader:
    inputs, labels = data
    print(inputs)
    print(labels)
for data in test_loader:
    inputs, labels = data
    print(inputs)
    print(labels)
'''

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 18 * 18, 800)
        self.fc2 = nn.Linear(800, 120)
        self.fc3 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 18 * 18)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# net = Net()
#
# #训练
# cirterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.5)
# for epoch in range(50):
#     running_loss = 0.0
#     for i, data in enumerate(train_loader, 0):
#         inputs, labels = data
#         inputs, labels = Variable(inputs), Variable(labels)
#         optimizer.zero_grad()                        # 优化器清零
#         outputs = net(inputs)
#         loss = cirterion(outputs, labels)
#         loss.backward()
#         optimizer.step()                         #优化
#         running_loss += loss.item()
#         if i % 200 == 199:
#             print('[%d %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
#             running_loss = 0.0
#
# print('finished training!')
#
# #测试
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in test_loader:
#         images, labels = data
#         outputs = net(Variable(images))
#         _, predicted = torch.max(outputs.data, dim=1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum()
# print('Accuracy of the network on the 400 test images: %d %%' % (100 * correct / total))
