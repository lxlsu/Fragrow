以LSTM文件夹为例：  
----model.png: 模型结构  
----r1：实验结果  
--------parameters.txt:输入的超参数以及loss输出  
--------evaluate.txt:预测结果的一些指标  
--------encoder.pth/decoder.pth：训练得到模型  
--------LSTMresults: 模型预测结果  
----results：在本地运行py，结果会生成在results文件夹   
Evaluate文件夹：计算预测结果的各项评估指标（其实里面大多数指标没有用到）  
训练数据：  
----不加蛋白质的数据：0507ligand_pro_15cutways.txt  
---------第一列是完整分子的SMILES序列，第二列是裁切后分子的SMILES序列。 
----加入蛋白质的数据：0525ligand_pro_15cutways.txt  
---------第一列是完整分子的SMILES序列，第二列是裁切后分子的SMILES序列，第三列是蛋白质序列。 
另外，LSTM+attention 和BiLSTM+attention两个模型没有单独画图，结构就是普通LSTM/BLSTM模型加入attention机制，训练数据都是没有加入蛋白质的数据。  
T-SNE可视化算法我实现的方法有点蠢有点麻烦，需要对数据集做很多操作，所以就先不放在里面了，师姐如果需要的话再问我要～  
