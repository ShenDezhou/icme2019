# icme2019
短视频内容理解与推荐竞赛

#战队成员  
chdwwk
yufang
*dezhou
levio


#Author  
ShenDezhou

#Version  
8.0

#入口  
automodel.py(legacy and golden entry, Track2 AUC=0.72(100%data), Track1 AUC=0.70（10%data）)  
automodel_batch.py(8.0 testing 7.0 AUC=0.52 6.0 AUC=0.52)  
train.py(baseline entrypoint, Good place to start)  


#changelog  
1.0使用前10000条  
2.0使用每10000条前100条  
3.0使用全量track2记录，但由于GPU内存不足，xDeepFM的embedding使用了1个  
4.0使用1%track1数据，2758600 records, 50 epochs, loss: 0.146  
5.0使用10%track2数据，PNN算法跑到0.68，目前的SOTA算法  
6.0增加新入口，使用batch模式来训练模型，待验证。  
7.0针对6.0分块训练结果不理想的问题，重新修改了逻辑，基本跟automodel.py一样，只是保留了一些enhancement，比如sparse feature分析的代码。  
7.1我的850MGPU只有1.9G，用Track2全量时，模型太大无法装入显存，于是把PNN模型的embedding_size改成1，ROC——AUC只有0.54，这样看来使用大内存更好。
8.0在Track1上运行算法，使用10%数据集，运行4Epochs就收敛停止了，0.70445 (0.65,0.84)，RANK#35  
8.1在Track2下使用100%数据集，由于GPU显存原因Embedding Size=2，训练结束Score=0.51，入口代码使用automode_batch.py,需要更大的显存重新跑这一版

#准备  
创建input文件夹  
Track1、TRACK2  
https://pan.baidu.com/s/1YDJ1yRy3m0KvlvTKJ7_jdg
提取码：6v34

#训练  
代码中使用了10000行数据进行100轮训练，该训练集上的结果为AUC1.0 1.0

#成绩  
0.576847387841962
track2: f, l = [0.56782278397016517, 0.59790479687615328]  
0.623347830227737
track2: f, l = [0.60371408494758505, 0.6691599025480921]  
0.724962529711026
track2: f, l = [0.65664805622285116, 0.88436296785010238]  


#Rank  
95     dezhou 0.57685 (0.57,0.60)	1  
130	levio 	0.72496 (0.66,0.88)	3  
225    tpt    0.72496 (0.66,0.88)	6  

#哈哈，好low啊

##下面是抖音公司的Baseline，用训练集跑出来的结果应该是finish-score, like-score = 0.70671501437, 0.920829590357

## 方案说明
- 特征：均为原始特征，不包含多媒体内容特征。使用到的特征字段 ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel',
       'music_id', 'did',]
- 模型：基于xDeepFM简单修改的多任务模型(没有测过开预测的效果，也可能分开做更好)。
- 结果：track2:  0.77094938716636 f, l = 0.70671501437, 0.920829590357

## 运行环境

 python 3.6  
 deepctr==0.3.1 
 tensorflow-gpu(tensorflow)
 pandas
 scikit-learn

### deepctr安装说明
- CPU版本
  ```bash
  $ pip install deepctr==0.3.1
  ``` 
- GPU版本
  先确保已经在本地安装`tensorflow-gpu`,版本为 **`tensorflow-gpu>=1.4.0,!=1.7.*,!=1.8.*,<=1.12.0`**，然后运行命令
    ```bash
    $ pip install deepctr==0.3.1 --no-deps
    ```


## 运行说明
1. 将track2对应的数据下载并解压至`input`目录内
2. 根据离线测试和线上提交修改`train.py`中的`ONLINE_FLAG`变量，运行`train.py`文件

