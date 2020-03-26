# LSTM
A LSTM based Chinese Segment Project.

## BiLSTM中文分词模型
在ICWS2005-PKU语料下训练精度达到99.99%，测试集上精度94.34%，召回94.21%, F1-值94.26%。

## 测试日志
单元测试及性能测试：  
CPU：Intel i56300HQ 2.30Ghz  
SSD: Samsung 970 EVO 1TB M.2 NVMe PCIe SSD  
GPU：GeForce GTX 950M-DDR3   
字典加载时间：176ms  
模型及权重加载时间：1m45s664ms  
推理性能：  
47.47ms/字 #以"我 昨天 去 清华 大学 。他 明天 去 北京 大学 ， 再 后天 去 麻省 理工大学 。"为测试条件；  
13.30ms/行 #以PKUTEST1944行为测试条件。  


2020-03-26 12:16:10,439 create pub-bilstm-bn:  
unigram dict:4698  
bigram dict:277325  
2020-03-26 12:16:10,615 load keras model:  
2020-03-26 12:17:56,279 inference:  
name: GeForce GTX 950M major: 5 minor: 0 memoryClockRate(GHz): 1.124  
__________________________________________________________________________________________________
|Layer (type)                    |Output Shape         |Param #     |Connected to|                     
|----------- |----------- |----------- |-----------|
|input_1 (InputLayer)            |(None, 1019, 5)      |0          |                                  |
|lambda_1 (Lambda)               |[(None, 1019), (None |0           |input_1[0][0]                    |
|embedding_1 (Embedding)         |(None, 1019, 100)    |469900      |lambda_1[0][0]                   |
|embedding_2 (Embedding)         |(None, 1019, 100)    |469900      |lambda_1[0][1]                   |
|embedding_3 (Embedding)         |(None, 1019, 100)    |469900      |lambda_1[0][2]                   |
|maximum_1 (Maximum)             |(None, 1019, 100)   | 0           |embedding_1[0][0]  embedding_2[0][0]                |
|maximum_2 (Maximum)            | (None, 1019, 100)    |0           |embedding_3[0][0]  embedding_2[0][0]                
|embedding_4 (Embedding)         |(None, 1019, 100)    |27732600    |lambda_1[0][3]                   |
|embedding_5 (Embedding)         |(None, 1019, 100)    |27732600    |lambda_1[0][4]                   |
|concatenate_1 (Concatenate)     |(None, 1019, 500)    |0           |embedding_1[0][0]   maximum_1[0][0]   maximum_2[0][0]    embedding_4[0][0]    embedding_5[0][0]                |
|spatial_dropout1d_1 (SpatialDro |(None, 1019, 500)    |0           |concatenate_1[0][0]              |
|bidirectional_1 (Bidirectional) |(None, 1019, 150)    |782400      |spatial_dropout1d_1[0][0]        |
|batch_normalization_1 (BatchNor |(None, 1019, 150)    |600         |bidirectional_1[0][0]          |  
|dense_1 (Dense)                 |(None, 1019, 4)      |604         |batch_normalization_1[0][0]   |   

Total params: 57,658,504  
Trainable params: 57,658,204  
Non-trainable params: 300  
__________________________________________________________________________________________________
2020-03-26 12:17:56,279 inference:  
2020-03-26 12:17:58,415 inference done.  
['我 昨天 去 清华 大学 。', '他 明天 去 北京 大学 ， 再 后天 去 麻省 理工大学 。']  
2020-03-26 12:17:58,415 inference:  
2020-03-26 12:17:58,415 inference pkutest:  
2020-03-26 12:18:24,262 inference pkutest done.  

##3. Installation/安装
3.1 Install dependencies, for convenience use `numpy==1.18.1, keras==2.2.4, tensorflow-gpu==1.15.2` for this case.    
3.2 Download Pretrained-Unigram-Bigram-BiLSTM-BN Chinese Segement keras model, arch file and weights file are seperated due to model .  
`https://pan.baidu.com/s/1LnjZD9HVQ164uAe0-XpPsg`, extract code:`zm41`  or use barcode to download the model.  
3.3 Clone project code.`git clone https://github.com/ShenDezhou/LSTM`

##4. Usage/使用说明
4.1 Create an object of PUB_BiLSTM_BN class.  
4.2 Specify dic file and model file using commandline argument `-u <unigramfile> -b <bigramfile> -a <archfile> -w <weightfile>`.
By default parameters are as follows:
```
UNIGRAM = 'pku_dic/pku_dict.utf8'  #字典
BIGRAM = 'pku_dic/pku_bigram.utf8'  #二字词典
MODELARCH = 'keras/B20-E60-F5-PU-Bi-Bn-De.json'  #keras模型
MODELWEIGHT = "keras/B20-E60-F5-PU-Bi-Bn-De-weights.h5"  #keras权重
```
4.3 Demo for making segments.
```
bilstm = PUB_BiLSTM_BN()
bilstm.loadKeras()
segs = bilstm.cut(["我昨天去清华大学。", "他明天去北京大学，再后天去麻省理工大学。"])
```
 
##5. 准确率及性能
在ICWS2005-PKU语料下训练精度达到99.99%，测试集上精度94.34%，召回94.21%, F1-值94.26%。  


模型加载性能、推理性能：  
CPU：Intel i56300HQ 2.30Ghz  
SSD: Samsung 970 EVO 1TB M.2 NVMe PCIe SSD  
GPU：GeForce GTX 950M-DDR3   
字典加载时间：176ms  
模型及权重加载时间：1m45s664ms  
推理性能：  
47.47ms/字 #以"我 昨天 去 清华 大学 。他 明天 去 北京 大学 ， 再 后天 去 麻省 理工大学 。"为测试条件；  
13.30ms/行 #以PKUTEST1944行 为测试条件。  

##6.  与Jieba、IK、CRF、Stanza等分词性能的比较
将jieba、IK、pkuseg/CRF、BiLSTM以及Stanza的评价进行对比，比较各模型的Macro评价，宏平均是对各类别的评价指标求算数平均值：
![image](https://pic4.zhimg.com/80/v2-c187acba2d33dc008ca7d5cd6f5e9e87_1440w.jpg)

##7. 结论
本文提出了一种基于预训练字与二字向量的BiLSTM中文分词模型，其性能取得了超过同类分词模型的效果。