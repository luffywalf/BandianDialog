#bandian_entity
name,id,class_num中 classnum的得来就是和id相同，除非前面有出现过同名id即类别
此文件下的id 和 node_bandian_q_pos中的id是一致的，但类别不同的，下面也说了

#bandian_q_pos.csv 中的class_id类别 是直接在node_bandian_q_pos.csv的类别名找的 和bandian_entity.csv中的类别不是一回事，
 这应该是因为大部分叶节点都没有做cnn分类训练 没有加到语料中；觉得比如流程某些点用户不会主动问到。
 2.这个文件是由node_bandian_q_pos.csv来的 之所以没直接用node_bandian_q_pos.csv 主要是为了代码csv列名一致
 3.这个文件一般手动改就好了

 4.改动 对类别20，22，39，40分别添加了近50个语料 24类添加17个
   这样做的原因是 增加对机关公司的识别率（数据增强）但这样会造成语料不平衡 不知道会不会对分类造成影响
   现在一共757条

   分类时候 unk类原本表现的就不好
   原来的分类只是针对user_q$class_label做的对吗 没有std_q 没错～


# two class.csv 直接在6000条上面改实在太多了
我决定在6000条中随机抽出2000条 然后在上面改
替换部分 我是+我
替换 低压+高压 2+37/   企业+机关 - 20+22 39+40 20+40 22+39
替换 军队24 + 20／22／39／40
最后得到1918条
由于单实体的共2000左右 所以我这边用1500条把先 抽出来
# test [ham 0.00127866, sub_acc 0.936508, rank_pre 0.977778]