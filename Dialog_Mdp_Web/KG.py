#coding=utf-8
from treelib import Tree
import numpy as np
import copy
import pandas as pd

class KG:
    # 这部分该怎么写呢？先想想这部分都要用在什么地方上？
    # 1.互斥关系的实体表
    # 2.基本：要有一个整体的树，用来走流程的，当前走到哪一步那种的；在生成回复的时候，要用到遍历孩子节点
    # 消歧的时候，从class_name -> entity_name；消歧本身过程，也是需要那个树的结构
    # 我的对话历史 是用数组和链表来控制的
    # 以上, 存储一个树的结构，一个状态数组，一个状态链表，一个互斥实体表（这个其实可以没有，但是这样查询比较快）

    # 最近公共祖先节点（还有别的经典算法）

    def __init__(self):
        self.bandian_rel_path = "src/data/bandian_rel.csv"
        self.bandian_entity_path = "src/data/bandian_entity.csv"

    def build_tree(self):

        df_del = pd.read_csv(self.bandian_rel_path)
        df_en = pd.read_csv(self.bandian_entity_path)
        id_en_dict = {}
        for i in range(len(df_en)):
            assert df_en["id"][i] not in id_en_dict
            id_en_dict[int(df_en["id"][i])] = df_en["name"][i]

        tree = Tree()
        tree.create_node("用户", 1)  # root node
        for i in range(len(df_del)):
            tree.create_node(id_en_dict[df_del["to_id"][i]], int(df_del["to_id"][i]), parent=int(df_del["from_id"][i]))

        # tree.create_node("用户", 0)  # root node
        # tree.create_node("高压", 1, parent=0)
        # tree.create_node("流程", 2, parent=1)
        # tree.create_node("材料", 3, parent=2)
        # tree.create_node("低压", 4, parent=0)
        # tree.create_node("居民", 5, parent=4)
        # tree.create_node("流程", 6, parent=5)
        # tree.create_node("材料", 7, parent=6)
        # tree.create_node("客户有效身份证明", 8, parent=7)
        # tree.create_node("身份证", 9, parent=8)
        # tree.create_node("护照", 10, parent=8)
        # tree.create_node("房产证", 11, parent=7)
        # tree.create_node("非居民", 12, parent=4)
        # tree.create_node("流程", 13, parent=12)
        # tree.create_node("材料", 14, parent=13)
        # tree.create_node("主体证明", 15, parent=3)

        return tree

    def get_latest_common_ancestor(self, tree, ids):
        # return ancestor's id
        # 对多个点的消歧
        # 不应该处理输入为[x, x's father]的情况

        if len(ids) <= 1:
            raise ValueError('length of ids less than 1')

        for i in range(len(ids)):
            for j in range(len(ids)):
                if i != j:
                    if tree.is_ancestor(ids[i], ids[j]):
                        raise ValueError('there is no disambiguation in ids')
        # NLU消歧部分需要
        ids_init = copy.deepcopy(ids)

        # depth list of ids
        ds = []
        for id in ids:
            ds.append(tree.depth(id))

        assert len(ds) == len(ids)

        d_min = min(ds)
        id_min = np.argmin(ds)

        ids_new = []
        for ith, d in enumerate(ds):
            for i in range(d - d_min):
                if ith == id_min:
                    break
                else:
                    ids[ith] = tree.parent(ids[ith]).identifier
            ids_new.append(ids[ith])

        assert len(ids) == len(ids_new)

        # same depth now;
        for i in range(d_min):
            for j in range(len(ids_new)):
                ids_new[j] = tree.parent(ids_new[j]).identifier

            flag = 0
            h = 0
            for h in range(len(ids_new)-1):
                if ids_new[h] != ids_new[h+1]:
                    flag = 1
                    break
            if flag == 0:
                # 最慢到根节点也会停下来
                return ids_init, ids_new[h]


if __name__ == '__main__':

    kg = KG()
    tree = kg.build_tree()
    # lca_id = kg.get_latest_common_ancestor(tree, [9,7,8])
    # print(lca_id)




