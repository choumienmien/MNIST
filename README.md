# MNIST_Myself DataSet
## 說明
利用  Pytorch實作 Machine Learning 演算法 - 神經網路，辨識手寫數字

## 資料集（Dataset）
[主要資料庫](http://pan.baidu.com/s/1pLMV4Kz "懸停顯示")
並加入自己創建的手寫數字，分別放入訓練集(驗證集)以及測試集

## 網路說明

#### 網路架構（Neural Network Architecture）
- 使用PyTorch框架創建一個帶有3層全連接網絡模型  
- 損失函數:交叉熵(CrossEntropyLoss)
- 最佳化器 (Optimizer) ：Adam()

#### 驗證模型(Cross-Validation)
使用 K-Fold 方法進行測試資料劃分
![image](https://github.com/choumienmien/MNIST/assets/37107594/2bd3bec5-8f76-4f63-9b92-ccba39bbc634)
[資料來源](https://ithelp.ithome.com.tw/articles/10279240)
- 資料分為「訓練階段(Train Phase)」以及「測試階段(Test Phase)」，
- 透過K-Fold將訓練階段的數據，分為訓練集(Training Set)與驗證集(Valid Set)
- 將訓練資料劃分成4、5、6個fold來看哪一個 fold 的準確率比較高，以作為測試階段的模型。

#### 參數設定
使用 CNN 模型


  
