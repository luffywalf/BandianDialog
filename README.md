#### 基于知识图谱的多轮对话系统 - 办电

##### 办电知识的原网页  
http://www.95598.cn/static/html//person/sas//PM06001003_997.shtml

##### bandian_dialogue下分三个文件夹，分别是  
* Dialogue_Mdp_Web(对话系统所在)
* MDP（得到DM需要的Q表）
* NLU（得到NLU需要的两个分类模型）  

##### 系统介绍 - 我的中期预答辩PPT   

##### 如何运行
1. 环境安装 python==3.6 以及 其他安装包见 requirements.txt (pip install -r requirements)
2. 进入MDP/Q-learning/文件夹，执行 python q-learning_goal.py  
   得到 MDP/data/Q_Table.npy 并放入 Dialog_Mdp_Web/src/generate/ （现在是已经放好的）
3. 进入NLU文件夹
   * 对于cnn-intent-classfication
     * 直接运行train.py 得到模型文件(cnn-intent-classification/data/runs_bandian/../checkpoints)和词表文件(在cnn-intent-classification/data/processor.vocab)
     * 将模型文件放到Dialog_Mdp_Web/NLU.py的self.intent_model_path 和 self.intent_vocab_pro
   * 对于cnn-text-classfication
     * 首先 下载自己想用的词向量（默认300维）到 Dialog_Mdp_Web/src/emb
     * 更改NLU/cnn-text-classification-tf/data_helpers.py 中的w2v_path 为Dialog_Mdp_Web/src/emb/某词向量
       非首次运行train.py的话 可以把data_helpers.py中的252行注释，用253和254 这样不用每次load词向量 能快点
     * 运行train.py 得到一个模型文件和两个词向量文件
     * 将其放入Dialog_Mdp_Web/NLU.py的self.text_model_path和 self.word2id_pkl_path, self.word_emb_path
4. 进入Dialogue_Mdp_Web 开始运行对话系统
   * 运行 python example.py 同时打开网页输入 127.0.0.1:9999 进入系统网页
     （也可以不在网页端运行，将example.py中的if_web设为false就可以 方便自己本地调试）
   
   
   
#### 树状图谱（部分）
![image](https://github.com/luffywalf/BandianDialog/blob/master/picture/kg_example.png)

#### 网页端样例
![image](https://github.com/luffywalf/BandianDialog/blob/master/picture/dialog_1.png)
![image](https://github.com/luffywalf/BandianDialog/blob/master/picture/dialog_2.png)
![image](https://github.com/luffywalf/BandianDialog/blob/master/picture/dialog_3.png)




