import torch
from torch.utils import data as data_
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as tranforms
import pylab

# 自動下載資料集
data_dir = './fashion_mnist/'
tranform = tranforms.Compose([tranforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(data_dir, train=True, transform=tranform,download=True)


# hyperparameter
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

# 查看MNIST照片
val_dataset= torchvision.datasets.MNIST(root=data_dir, train=False, transform=tranform)
print('訓練資料集筆數:',len(train_dataset))
print('測試資料集筆數:',len(val_dataset))
im = train_dataset[0][0]
im = im.reshape(-1,28)
pylab.imshow(im)
pylab.show()
print("該圖片的標籤為：",train_dataset[0][1])

print(train_dataset.data.size()) #每張大小是28乘28。
print(train_dataset.targets.size())
plt.ion()
for i in range(11):
  plt.imshow(train_dataset.train_data[i].numpy(), cmap = 'gray')
  plt.title('Pic_lable %i' % train_dataset.targets[i])
  plt.pause(0.5)
plt.show()