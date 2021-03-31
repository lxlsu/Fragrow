以LSTM文件夹为例：  
  model.png: 模型结构  
  LSTMresults: 模型预测结果  
  results：在本地运行py，结果会生成在results文件夹   
Evaluate文件夹：计算预测结果的各项评估指标（其实里面大多数指标没有用到）  
训练数据：  
  不加蛋白质的数据：0507ligand_pro_15cutways.txt  
  第一列是完整分子的SMILES序列，第二列是裁切后分子的SMILES序列。 
  加入蛋白质的数据：0525ligand_pro_15cutways.txt  
  第一列是完整分子的SMILES序列，第二列是裁切后分子的SMILES序列，第三列是蛋白质序列。 

