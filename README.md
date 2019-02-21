# icme2019
短视频内容理解与推荐竞赛

#Author
ShenDezhou

#Version
1.0

#准备
创建input文件夹
训练集地址：
http://lf1-ttcdntos.pstatp.com/obj/icme2019&bytedance_challenge_dataset/final_track2_train.txt.tgz
测试集地址：
http://lf1-ttcdn-tos.pstatp.com/obj/icme2019&bytedance_challenge_dataset/final_track2_test_no_anwser.txt.tgz

#训练
代码中使用了10000行数据进行100轮训练，该训练集上的结果为AUC1.0 1.0

#成绩
finish-score, like-score = [0.56782278397016517, 0.59790479687615328]

#Rank
95	
dezhou
0.57685 (0.57,0.60)	1

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

