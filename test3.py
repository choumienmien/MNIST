import csv
import math
import os
import random
import sys


import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.datasets.folder import default_loader
from sklearn.model_selection import KFold


# 將照片名稱寫入txt檔
def input_pic_txt():
    for num in range(10):
        name = num
        # img_path = './mnist/mnist/all/%s/' % name #訓練
        img_path = './mnist/mnist/test_myself_d/%s/' % name  # 測試
        print(img_path)
        # TxtName = './mnist/mnist/individual_txt_file/data%s .txt' % name  # 訓練
        TxtName = './mnist/mnist/individual_txt_test_file/data%s .txt' % name  # 測試
        f = open(TxtName, 'w')
        img_path_Line = os.listdir(img_path)
        for ImgName in img_path_Line:
            label = "%s" % name  # 修改你標簽的名字
            Name_Label = img_path + ImgName + ',' + label
            print(Name_Label)
            f.write(Name_Label)  # 格式化寫入也可以
            f.write('\n')  # 顯示寫入換行
        f.close()


# 將所有照片0-9寫入的tx，製作all.txt
def write_txt():
    # -*- coding:UTF-8 -*-
    # fq = open('./mnist/mnist/all.txt', 'w')  # 這裡用追加模式 #訓練
    fq = open('./mnist/mnist/test_m.txt', 'w')  # 測試
    for i in range(10):
        name = i
        # TxtName = './mnist/mnist/individual_txt_file/data%s .txt' % name  #訓練
        TxtName = './mnist/mnist/individual_txt_test_file/data%s .txt' % name  # 測試
        fp = open(TxtName, 'r')
        for line in fp:
            fq.write(line)
        fp.close()
    fq.close()


# -*- coding:utf-8 -*-
# 在txt文件中隨機抽取行
def split_train_test():
    All = open('./mnist/mnist/all.txt', 'r', encoding='utf-8')  # 要被抽取的文件All.txt，共63,131行
    trainf = open('./mnist/mnist/train.txt', 'w', encoding='utf-8')  # 抽取的0.7倍寫入train.txt
    testf = open('./mnist/mnist/test.txt', 'w', encoding='utf-8')  # 抽取的0.3倍寫入test.txt
    AllNum = 69166  # 總圖像數
    SetTrainNum = 0.9  # 設置比例
    SetTestNum = 0.1

    trainNum = math.floor(SetTrainNum * AllNum)
    testNum = math.floor(SetTestNum * AllNum)

    trainresultList = random.sample(range(0, AllNum), trainNum)  # sample(x,y)函數的作用是從序列x中，隨機選擇y個不重複的元素
    testresultList = random.sample(range(0, AllNum), testNum)  # sample(x,y)函數的作用是從序列x中，隨機選擇y個不重複的元素

    lines = All.readlines()
    for i in trainresultList:
        trainf.write(lines[i])
    trainf.close()
    for i in testresultList:
        testf.write(lines[i])
    testf.close()
    All.close()


#
# pass
# #定義讀取文件的格式


class MnistDataset(Dataset):
    def __init__(self, image_path, image_label, transform=None, target_transform=None,
                 loader=default_loader):  # __init__是初始化該類的一些基礎參數
        super(MnistDataset, self).__init__()
        self.image_path = image_path  # 初始化圖像路徑列表
        self.image_label = image_label  # 初始化圖像標簽列表
        self.transform = transform  # 初始化數據增強方法
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        根據索引index返回dataset[index]
        獲取對應index的圖像，並視情況進行數據增強
        """
        # image = Image.open(self.image_path[index]).convert("RGB")  # Convert image to RGB mode if needed # 讀取圖像
        # # image = np.array(image)
        # label = float(self.image_label[index])
        img_name = self.image_path[index]
        label = self.image_label[index]
        image = self.loader(img_name)

        if self.transform is not None:
            image = self.transform(image)  # 歸一化到 [0.0,1.0]
        if self.target_transform is not None:
            label = self.target_transform(image)
        return image, label

    def __len__(self):  # 返回整個數據集的大小
        return len(self.image_path)


# 全連接網絡層 + 激活層 + BN 網絡層

# 定義分類模型


class simpleNet(nn.Module):
    """
    定義了一個簡單的三層全連接神經網絡，每一層都是線性的
    """

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
    # """
    #     定義了一個簡單的三層全連接神經網絡，每一層都是線性的
    # """
    #     super(simpleNet, self).__init__()
    #     self.layer1 = nn.Linear(in_dim, n_hidden_1)
    #     self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
    #     self.layer3 = nn.Linear(n_hidden_2, out_dim)
        super(simpleNet, self).__init__()
        # 在上面的Activation_Net的基礎上，增加了一個加快收斂速度的方法——批標準化
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


# defining the training function
# Train baseline classifier on clean data
def train(model, optimizer, train_loader, criterion, epoch):
    model.train()  # setting up for training
    for batch_idx, (data, target) in enumerate(
            train_loader):  # data contains the image and target contains the label = 0/1/2/3/4/5/6/7/8/9
        data = data.view(-1, 3 * 28 * 28).requires_grad_()
        optimizer.zero_grad()  # setting gradient to zero# 清除梯度
        output = model(data)  # forward
        loss = criterion(output, target)  # loss computation
        loss.backward()  # back propagation here pytorch will take care of it
        optimizer.step()  # updating the weight values
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


# to evaluate the model
## validation of test accuracy
def val(model, criterion, val_loader, epoch, train=False):
    model.eval()  # 在eval模式下，dropout層會讓所有的激活單元都通過，而BN層會停止計算和更新mean和var，直接使用在訓練階段已經學出的mean和var值。
    val_loss = 0
    correct = 0

    with torch.no_grad():  # 不記錄模型梯度信息，以起到加速和節省顯存的作用。它的作用是將該with語句包裹起來的部分停止梯度的更新
        for batch_idx, (data, target) in enumerate(val_loader):
            data = data.view(-1, 3 * 28 * 28).requires_grad_()
            output = model(data)
            val_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[
                1]  # get the index of the max log-probability， #通過predict和標簽進行對比(predict是第幾位是最大的概率, 標簽是0 - 9的數字, 所以當predict和labels相等時就相當於預測值是正確的)來得到判斷正確的數量並賦給correct
            correct += pred.eq(target.view_as(pred)).sum().item()  # if pred == target then correct +=1

    val_loss /= len(val_loader.dataset)  # average val loss
    if train == False:
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            val_loss, correct, val_loader.sampler.__len__(),
            100. * correct / val_loader.sampler.__len__()))
    if train == True:
        print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            val_loss, correct, val_loader.sampler.__len__(),
            100. * correct / val_loader.sampler.__len__()))
    return 100. * correct / val_loader.sampler.__len__()


def test(save_path, model, criterion, test_loader):
    if os.path.exists(save_path):
        loaded_paras = torch.load(save_path)
        print("#### 成功載入已有模型，進行追加訓練...")
        # 打印參數
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        # 載入模型
        model.load_state_dict(loaded_paras)

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data = data.view(-1, 3 * 28 * 28).requires_grad_()
                output = model(data)
                test_loss += criterion(output, target).item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                print('=================')
                print(pred)
                print('=================')
                correct += pred.eq(target.view_as(pred)).sum().item()  # if pred == target then correct +=1

        test_loss /= len(test_loader.dataset)  # average test loss
        print('\ntest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss, correct, test_loader.sampler.__len__(),
            100. * correct / test_loader.sampler.__len__()))

        print(100. * correct / test_loader.sampler.__len__())
        # # 輸出原始圖

        print('num_of_testData:', len(test_dataset))
        # for i, (batch_x, batch_y) in enumerate(test_loader):
        #     print(i, batch_x.size(), batch_y.size())
        #     show_batch(batch_x)
        #     plt.axis('off')
        #     plt.show()
        for i in test_loader:
            img, label = i
            print(img.size(), label)
        sys.exit('測試資料輸出結束')

    else:
        print('需要訓練資料要繼續訓練')


def train_flod_Mnist(k_split_value, BATCH_SIZE, Iterations):
    different_k_mse = []
    kf = KFold(n_splits=k_split_value, shuffle=True,
               random_state=0)  # init KFold，設置shuffle=True和random_state=整數，每次運行結果相同
    # split幾次與epochs數值相乘就是，小計訓練幾次，之後再看循環幾個split
    for train_index, val_index in kf.split(train_dataset):  # split
        # get train, val
        train_fold = torch.utils.data.dataset.Subset(train_dataset, train_index)
        val_fold = torch.utils.data.dataset.Subset(train_dataset, val_index)

        # package type of DataLoader
        train_loader = DataLoader(dataset=train_fold, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(dataset=val_fold, batch_size=BATCH_SIZE, shuffle=True)

        # train model
        val_acc = torch.zeros([Iterations])
        train_acc = torch.zeros([Iterations])

        ## training the logistic model
        for i in range(Iterations):  # Iterations=epochs
            train(model, optimizer, train_loader, criterion, i)
            train_acc[i] = val(model, criterion, train_loader, i, train=True)  # validating the  current CNN
            val_acc[i] = val(model, criterion, val_loader, i)
            torch.save(model.state_dict(), 'perceptron{}.pt'.format(k_split_value))  # 保存模型
        # one epoch, all acc
        different_k_mse.append(np.array(val_acc))
    return different_k_mse


def get_path_label(label_file_path):
    """
    獲取數字圖像的路徑和標簽並返回對應的列表
    @ img_root:保存圖像的根目路
    @ label_file_path 保存圖像標簽的文件路徑，。cvs 或 。txt
    @ return：圖像的路徑列表和對應的標簽列表
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


if __name__ == '__main__':
    # ----------------------------------只要一次生成自己的數據庫----------------------------------
    # input_pic_txt()
    # write_txt()
    # split_train_test()
    # exit()
    # -------------------------------------超參數定義-------------------------------------
    batch_size = 64  # 一個batch的size
    learning_rate = 0.01
    num_epoches = 7  # 總樣本的迭代次數
    fold_num = 4

    # -------------------------------------選擇模型--------------------------------------
    model = simpleNet(3 * 28 * 28, 300, 100, 10)
    model.apply(init_weights)
    # print(model)

    # -------------------------------------定義損失函數和優化器--------------------------------------
    # -------------------------------------交叉熵和SGD優化器-------------------------------------
    criterion = nn.CrossEntropyLoss()  # softmax與交叉熵一起
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # -------------------------------------測試資料-------------------------------------

    save_path = "C:/Users/mandy chou/Desktop/MNIST/version4_2/perceptron.pt"
    # 獲取測試集路徑列表和標簽列表
    # 測試模型
    test_label = './mnist/mnist/test_m.txt'  # 輸入測試集的txt
    test_img_list, test_label_list = get_path_label(test_label)
    # 訓練集dataset
    test_dataset = MnistDataset(test_img_list, test_label_list, transform=transforms.Compose([transforms.ToTensor()]))
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    # #
    # total_step = len(test_loader)
    # print(total_step)
    test(save_path, model, criterion, test_loader)

    # -------------------------------------訓練與驗證-------------------------------------
    # 訓練與驗證
    # 獲取訓練集路徑列表和標簽列表
    train_label = './mnist/mnist/train.txt'  # 輸入訓練集的txt
    train_img_list, train_label_list = get_path_label(train_label)
    # 訓練和測試集預處理

    train_dataset = MnistDataset(train_img_list, train_label_list,
                                 transform=transforms.Compose([transforms.ToTensor()]))  # 訓練集dataset數據預處理方法

    # # 開始加載自己的數據
    # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
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

    # 設定K值為[2,10]進行訓練
    valAcc_compare_map = {}

    for k_split_value in range(fold_num, fold_num + 2 + 1):
        print('now k_split_value is:', k_split_value)
        valAcc_compare_map[k_split_value] = train_flod_Mnist(k_split_value, batch_size, num_epoches)

    dict = {}
    for k in valAcc_compare_map:  # k代表fold下的結果，分別是fold=4的array與fold=5的array k只會是1,2
        # for j in range(k):  # j代表fold下的每次訓練與驗證結果，內容會是epcho，如果是fold4，則代表j會是0,1,2,3
        #     print(valAcc_compare_map[k][j])
        print('------------------')
        arang_acc = float(np.mean(valAcc_compare_map[k]))
        dict_new = {fold_num: arang_acc}
        dict.update(dict_new)
        fold_num += 1
    with open('dct.csv', 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['fold', 'avg_acc'])
        for k, v in dict.items():
            writer.writerow([k, v])

