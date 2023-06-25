import os
import random
import math

import pandas as pd
import torch
from torch import nn, optim
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.datasets.folder import default_loader
from torch.autograd import Variable
from torch.nn import functional as F



# 將照片名稱寫入txt檔
def input_pic_txt():
    for num in range(10):
        name = num
        img_path = './mnist/mnist/all/%s/' % name
        print(img_path)
        TxtName = './mnist/mnist/data%s .txt' % name
        f = open(TxtName, 'a')
        img_path_Line = os.listdir(img_path)
        for ImgName in img_path_Line:
            label = "%s" % name  # 修改你标签的名字
            Name_Label = img_path + ImgName + ',' + label
            print(Name_Label)
            f.write(Name_Label)  # 格式化写入也可以
            f.write('\n')  # 显示写入换行
        f.close()


# 將所有照片0-9寫入的tx，製作all.txt
def write_txt():
    # -*- coding:UTF-8 -*-
    fq = open('./mnist/mnist/all.txt', 'a')  # 这里用追加模式
    for i in range(10):
        name = i
        TxtName = './mnist/mnist/data%s .txt' % name
        fp = open(TxtName, 'r')
        for line in fp:
            fq.write(line)
        fp.close()
    fq.close()


# -*- coding:utf-8 -*-
# 在txt文件中随机抽取行
def split_train_valid():
    All = open('./mnist/mnist/All.txt', 'r', encoding='utf-8')  # 要被抽取的文件All.txt，共63,131行
    trainf = open('./mnist/mnist/train.txt', 'w', encoding='utf-8')  # 抽取的0.7倍写入train.txt
    testf = open('./mnist/mnist/test.txt', 'w', encoding='utf-8')  # 抽取的0.3倍写入test.txt
    AllNum = 63131  # 总图像数
    SetTrainNum = 0.7  # 设置比例
    SetvalidNum = 0.3

    trainNum = math.floor(SetTrainNum * AllNum)
    validNum = math.floor(SetvalidNum * AllNum)

    trainresultList = random.sample(range(0, AllNum), trainNum)  # sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素
    validresultList = random.sample(range(0, AllNum), validNum)  # sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素

    lines = All.readlines()
    for i in trainresultList:
        trainf.write(lines[i])
    trainf.close()
    for i in validresultList:
        testf.write(lines[i])
    testf.close()
    All.close()


# split_train_valid()
# pass
# #定义读取文件的格式


class MnistDataset(Dataset):
    def __init__(self, image_path, image_label, transform=None, target_transform=None,
                 loader=default_loader):  # __init__是初始化该类的一些基础参数
        super(MnistDataset, self).__init__()
        self.image_path = image_path  # 初始化图像路径列表
        self.image_label = image_label  # 初始化图像标签列表
        self.transform = transform  # 初始化数据增强方法
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        根据索引index返回dataset[index]
        获取对应index的图像，并视情况进行数据增强
        """
        # image = Image.open(self.image_path[index]).convert("RGB")  # Convert image to RGB mode if needed # 读取图像
        # # image = np.array(image)
        # label = float(self.image_label[index])
        img_name = self.image_path[index]
        label = self.image_label[index]
        image = self.loader(img_name)

        if self.transform is not None:
            image = self.transform(image)  # 归一化到 [0.0,1.0]
        if self.target_transform is not None:
            label = self.target_transform(image)
        return image, label

    def __len__(self):  # 返回整个数据集的大小
        return len(self.image_path)


# 全连接网络层 + 激活层 + BN 网络层

#定义分类模型


class simpleNet(nn.Module):
    """
    定义了一个简单的三层全连接神经网络，每一层都是线性的
    """
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(simpleNet, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x



def get_path_label(label_file_path):
    """
    获取数字图像的路径和标签并返回对应的列表
    @ img_root:保存图像的根目路
    @ label_file_path 保存图像标签的文件路径，。cvs 或 。txt
    @ return：图像的路径列表和对应的标签列表
    """
    data = pd.read_csv(label_file_path, names=['img', 'label'])
    print(data)
    pass
    data['img'] = data['img'].apply(lambda x: x)

    return data['img'].tolist(), data['label'].tolist()


def show_batch(imgs):
    grid = utils.make_grid(imgs)  # 將批次圖片內容組合生成一個圖片
    plt.imshow(grid.numpy().transpose((1, 2, 0)))  # 將圖片轉置
    plt.title('Batch from dataloader')

# def fit_model(model, loss_func, optimizer, input_shape, num_epochs, train_loader, test_loader):
#     # Traning the Model
#     #history-like list for store loss & acc value
#     training_loss = []
#     training_accuracy = []
#     validation_loss = []
#     validation_accuracy = []
#     for epoch in range(num_epochs):
#         #training model & store loss & acc / epoch
#         correct_train = 0
#         total_train = 0
#         for i, (images, labels) in enumerate(train_loader):
#             # 1.Define variables
#             train = Variable(images.view(input_shape))
#             labels = Variable(labels)
#             # 2.Clear gradients
#             optimizer.zero_grad()
#             # 3.Forward propagation
#             outputs = model(train)
#             # 4.Calculate softmax and cross entropy loss
#             train_loss = loss_func(outputs, labels)
#             # 5.Calculate gradients
#             train_loss.backward()
#             # 6.Update parameters
#             optimizer.step()
#             # 7.Get predictions from the maximum value
#             predicted = torch.max(outputs.data, 1)[1]
#             # 8.Total number of labels
#             total_train += len(labels)
#             # 9.Total correct predictions
#             correct_train += (predicted == labels).float().sum()
#         #10.store val_acc / epoch
#         train_accuracy = 100 * correct_train / float(total_train)
#         training_accuracy.append(train_accuracy)
#         # 11.store loss / epoch
#         training_loss.append(train_loss.data)
#
#         #evaluate model & store loss & acc / epoch
#         correct_test = 0
#         total_test = 0
#         for images, labels in test_loader:
#             # 1.Define variables
#             test = Variable(images.view(input_shape))
#             # 2.Forward propagation
#             outputs = model(test)
#             # 3.Calculate softmax and cross entropy loss
#             val_loss = loss_func(outputs, labels)
#             # 4.Get predictions from the maximum value
#             predicted = torch.max(outputs.data, 1)[1]
#             # 5.Total number of labels
#             total_test += len(labels)
#             # 6.Total correct predictions
#             correct_test += (predicted == labels).float().sum()
#         #6.store val_acc / epoch
#         val_accuracy = 100 * correct_test / float(total_test)
#         validation_accuracy.append(val_accuracy)
#         # 11.store val_loss / epoch
#         validation_loss.append(val_loss.data)
#         print('Train Epoch: {}/{} Traing_Loss: {} Traing_acc: {:.6f}% Val_Loss: {} Val_accuracy: {:.6f}%'.format(epoch+1, num_epochs, train_loss.data, train_accuracy, val_loss.data, val_accuracy))
#     return training_loss, training_accuracy, validation_loss, validation_accuracy


if __name__ == '__main__':
     # -------------------------------------超参数定义-------------------------------------
    batch_size = 8  # 一个batch的size
    learning_rate = 0.02
    num_epoches = 3  # 总样本的迭代次数

    # 获取训练集路径列表和标签列表
    train_label = './mnist/mnist/train.txt'  # 输入训练集的txt
    train_img_list, train_label_list = get_path_label(train_label)
    # 训练和测试集预处理

    train_dataset = MnistDataset(train_img_list, train_label_list,
                                 transform=transforms.Compose([transforms.ToTensor()]))  # 训练集dataset数据预处理方法
    # 获取测试集路径列表和标签列表
    # test_data_root = ImgPath
    test_label = './mnist/mnist/test.txt'  # 输入测试集的txt
    test_img_list, test_label_list = get_path_label(test_label)
    # 训练集dataset
    test_dataset = MnistDataset(test_img_list, test_label_list, transform=transforms.Compose([transforms.ToTensor()]))
    # 开始加载自己的数据
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    # print('num_of_trainData:', len(train_dataset))
    # print('num_of_testData:', len(test_dataset))
    # for i, (batch_x, batch_y) in enumerate(train_loader):
    #     if (i < 4):
    #         print(i, batch_x.size(), batch_y.size())
    #         show_batch(batch_x)
    #         plt.axis('off')
    #         plt.show()

    # for i in train_loader:
    #     img, label = i
    #     print(img.size(), label)
    #-------------------------------------选择模型--------------------------------------

    model = simpleNet(3*28 * 28, 300, 100, 10)
    # print(model)

    # -------------------------------------定义损失函数和优化器--------------------------------------
    # 交叉熵和SGD优化器
    criterion = nn.CrossEntropyLoss()  # softmax与交叉熵一起
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # input_shape = (-1,1,28,28)
    # training_loss, training_accuracy, validation_loss, validation_accuracy = fit_model(model, criterion, optimizer, input_shape, num_epoches, train_loader, test_loader)

    print('Start Training!')
    iter = 0 #迭代次数
    for epoch in range(num_epoches):
        for data in train_loader:
            img, label = data
            img = img.view(img.size(0), -1)
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()
            else:
                img = img
                label = label
            out = model(img)
            loss = criterion(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter+=1
            #每迭代50次打印一次
            if iter%50 == 0:
                print('epoch: {}, iter:{}, loss: {:.4}'.format(epoch, iter, loss.data.item()))
