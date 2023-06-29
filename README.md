# MNIST_Myself DataSet
## 說明
利用  Pytorch實作 Machine Learning 演算法 - 神經網路，辨識手寫數字

## 資料集（Dataset）
[主要資料庫](http://pan.baidu.com/s/1pLMV4Kz "懸停顯示")
並加入自己創建的手寫數字，分別放入訓練集(驗證集)以及測試集
資料總長度為:
  -  訓練階段 (Train Phase) 筆數: 62,281
  -  測試階段 (Test Phase) 筆數: 6,955，其中至少有 35 筆是加入自己手寫的資料。
  - 35 筆資料如下:

     ![Figure_1](https://github.com/choumienmien/MNIST/assets/37107594/637253a6-c8c7-4898-97e7-dd7e5de0734a)


## 網路說明

#### 網路架構（Neural Network Architecture）
- 使用PyTorch框架創建一個帶有3層全連接網絡模型  
- 損失函數:交叉熵(CrossEntropyLoss)
- 最佳化器 (Optimizer) ：Adam()
- 使用卷積神經網絡（Convolutional Neural Network） 模型
 - 2 層捲積層 + 3 層全連接層 

  <img width="510" alt="image" src="https://github.com/choumienmien/MNIST/assets/37107594/176b5c7d-9591-4e96-8e9a-4e6e5651a01f">

  [圖片來源](https://medium.com/jameslearningnote/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC5-1%E8%AC%9B-%E5%8D%B7%E7%A9%8D%E7%A5%9E%E7%B6%93%E7%B6%B2%E7%B5%A1%E4%BB%8B%E7%B4%B9-convolutional-neural-network-4f8249d65d4f)

  ![image](https://github.com/choumienmien/MNIST/assets/37107594/92ab1d13-8bbc-46f3-8eed-b34113767d94)


#### 驗證模型(Cross-Validation)
使用 K-Fold 方法進行測試資料劃分
![image](https://github.com/choumienmien/MNIST/assets/37107594/155a25bd-a1e0-4825-b54e-6c610a4d2c7a)

[圖片來源](https://ithelp.ithome.com.tw/articles/10279240)

- 資料分為「訓練階段 ( Train Phase )」以及「測試階段 ( Test Phase )」，
- 透過K-Fold將訓練階段的數據，分為訓練集 ( Training Set ) 與驗證集 ( Valid Set )
- 將訓練資料劃分成4、5、6個fold來看哪一個 fold 的準確率比較高，以作為測試階段的模型。

#### 參數設定
- 批量大小 ( batch_size ) = 64 
- 學習率 ( learning_rate ) = 0.01
- 循環次數 ( num_epoches ) = 7
- 總分割數 ( fold_num ) = 4

## 結論
從下圖得知 fold: 5 整體的準確率比較高，但針對自己手寫數字進行測試，其準確率相較 fold_6 下降 8.57%，或許放入更多自己手寫的圖片進行訓練，會增加後續驗證的準確率。
![image](https://github.com/choumienmien/MNIST/assets/37107594/98429f8c-5cc3-41b6-ac43-c4871144d1f0)

  
## 資料來源
* [动手撸个自己的数据集进行训练Pytorch框架（索引式）](https://juejin.cn/post/7078130257970069518 "懸停顯示")
* [MNIST数据集的读取、显示以及全连接实现数字识别](https://blog.csdn.net/QLeelq/article/details/121069095 "懸停顯示")
* [MNIST集的测试(详细步骤)](https://blog.csdn.net/whale_ss/article/details/129939960?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-2-129939960-blog-123781738.235%5Ev38%5Epc_relevant_default_base3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-2-129939960-blog-123781738.235%5Ev38%5Epc_relevant_default_base3&utm_relevant_index=5 "懸停顯示")
* [PyTorch基础入门六：PyTorch搭建卷积神经网络实现MNIST手写数字识别](https://blog.csdn.net/out_of_memory_error/article/details/81434907 "懸停顯示")
* [貓狗數據集](https://www.cnblogs.com/xiximayou/p/12398285.html "懸停顯示")
* [万物皆用MNIST---MNIST数据集及创建自己的手写数字数据集](https://blog.csdn.net/m0_62128864/article/details/123781738 "懸停顯示")
