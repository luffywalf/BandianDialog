换词向量 - 300d
统一设置路径参数 - 没找到特别好的全局参数 就在每个文件init设置吧

词向量一直下载不下来 不想卡在这步 就先用已有的PKL了
所以 如果是别的及其要用的话 要先做如下的generate的东西
1.先下载词向量
2.分别run NLU下的两个分类模型得到模型文件 写入相应位置；得到intent的vocab文件 写入相应位置
3.run Q-learning得到Q表

还有环境配置install的那些还没弄 
2019/10/19
我跑了下intent模型没问题 text没跑 跑了Q-learning 应该没什么问题 先改逻辑和NLG吧
2019/10/20
先把NLU text换成单实体 
调试时候 词向量也可以改成不加 效果应该不会差很多 但是会快很多；暂时还没改 确认这边没问题了再改
现在就NLU还是有不准的 比如什么材料 - 流程 => 可以先弄NLG 最后改进NLU 逻辑现在暂时什么问题

NLG
1. 有些太生硬 
2. 多样性

1. 
申请受理包括：
材料
方案答复
 --- 变成 携带材料 然后会进行 方案答复
 
 材料包括：
主体证明
报装申请资料
项目立项及批复
--- 这里的材料包括...


后面改进
可以自己训练词向量 拿通用预料加上自己的办电语料


需要的数据

KG
/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/resources/database/bandian_rel.csv
/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/resources/database/bandian_entity.csv

web
../FSM_v1/chat_web/

data_helpers
/home/dn/WordEmbedding/sgns.weibo.word 好像不用非得用
/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/cnn-text-classification-tf/PKL/sgns.chinese/word2id.pkl
/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/cnn-text-classification-tf/PKL/sgns.chinese/word_embedding.npy
src/generate/PKL/word2id.pkl 两个生成的

DM
data/Q_Table.npy

NLU
/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/cnn-text-classification-tf/PKL/sgns.chinese/word2id.pkl
/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/cnn-text-classification-tf/PKL/sgns.chinese/word_embedding.npy
/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/cnn-text-classification-tf/runs_bandian/1554965249/checkpoints/
data/processor.vocab
/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/cnn-intent-classification/runs_bandian/1544259188/checkpoints
/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/resources/database/node_bandian_q_pos.csv



原有的数据
bandian_rel.csv
bandian_entity.csv
node_bandian_q_pos.csv
../FSM_v1/chat_web/
sgns.weibo.word 好像不用非得用 弄个小点的词向量估计也差不多


生成的数据
Q_Table.npy
PKL/sgns.chinese/word2id.pkl
PKL/sgns.chinese/word_embedding.npy
cnn-text-classification-tf/runs_bandian/1554965249/checkpoints/
cnn-intent-classification/runs_bandian/1544259188/checkpoints
data/processor.vocab


requirment.txt

kg
from treelib import Tree, Node
import numpy as np
import copy
import pandas as pd

web
from __future__ import print_function
from __future__ import absolute_import
import sys
import argparse

data_helpers
import numpy as np
import re
import jieba
import pandas as pd
from gensim.models import KeyedVectors

NLU
import tensorflow as tf
import pandas as pd
from tensorflow.contrib import learn
import jieba
import numpy as np
import time

