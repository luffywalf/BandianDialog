import KG
import pandas as pd
kg = KG.KG()
tree = kg.build_tree()

class NLG:
    nlg_answer_path = "src/data/nlg.csv"
    basic_user_info_set = {"低压", "高压", "居民", "非居民"}
    mutex_set = {"军队", "企业", "企业或工商客户", "机关事业单位或其他非营利组织", "机关事业单位"}
    mutex_set_id = set()
    for m in mutex_set:
        for n in tree.all_nodes():
            if n.tag == m:
                mutex_set_id.add(n.identifier)

    if_set = {"委托人", "私人宅基地建筑", "农村", "国家优惠价", "新建建筑物","国家优惠电价",
              "钢铁、电解铝、铁合金、水泥、电石、烧碱、黄磷、锌冶炼高耗能等特殊行业客户", "需要规划立项"}
    # if_two_set 也是条件，但是租赁这个同时需要本节点和子节点
    if_two_set = {"租赁"}
    user_set = mutex_set | if_set | if_two_set | basic_user_info_set

    # 条件类实体中，同时需要高低压评判和父节点评判的点的集合
    if_class_two = {"委托人的身份证明","委托授权书","政府或规划部门的批复文件","乡镇及以上政府或规划部门的批复文件",
                    "政府有关部门核发的资质证明和公益证明（学校、养老院、居委会）","立项、规划批复文件","营业执照复印件",
                    "团级及以上证明","事业法人证书或组织机构代码证复印件","事业法人证书或组织机构代码证复印件",
                    "由所在乡、镇或乡镇以上级别政府部门根据所辖权限开据证明","政府有关部门核发的资质证明和公益证明（学校、养老院、居委会等）",
                    "租赁协议复印件及产权人同意报装证明材料","相关政府部门批准的许可文件包括政府主管部门立项或批复文件，环境评估报告、生产许可证等",
                    "建设工程规划许可证及附件"}
    no_set = {"不", "非", "否"}

    def __init__(self):
        self.nlg_answer_array = []
        df_nlg_answer = pd.read_csv(self.nlg_answer_path)
        for i in range(len(df_nlg_answer)):
            self.nlg_answer_array.append(df_nlg_answer["sentence"][i])

    def check_if_no(self, sent):
        # TODO 可能还不够完善？
        for x in self.no_set:
            if x in sent:
                return True
        return False

    def rhetoric_generator(self, io_method, tree, nid, ids):
        # 消歧反问的句子生成规则
        children_list = tree.children(nid)
        io_method.out_fun("请问您属于：")#11
        for i in range(len(children_list)):
            and_set = set([x for x in tree.subtree(children_list[i].identifier).nodes]) & set(ids)
            if and_set:
                io_method.out_fun(children_list[i].tag)#11
        return children_list

    def answer_for_user(self, io_method, tree, nid, state_array, state_linklist):
        state_array[nid] = 2
        state_linklist.append(nid)

        ans_other = []

        children_list = tree.children(nid)
        io_method.out_fun("您需要：")#12
        for i in range(len(children_list)):
            io_method.out_fun(children_list[i].tag)#12

        parent = tree.parent(nid)
        if not parent.is_root and parent:
            ans_parent, state_array, state_linklist = self.answer_for_stuff(
                io_method, tree, parent.identifier, state_array, state_linklist)
            grandpa = tree.parent(parent.identifier)

            for a in ans_parent:
                for i in range(len(children_list)):
                    # 除掉含有已经说过的
                    if a.identifier != children_list[i].identifier:
                        ans_other.append(a)
            if grandpa:
                ans_grandpa, state_array, state_linklist = self.answer_for_stuff(
                    io_method, tree, grandpa.identifier, state_array, state_linklist)
                for a in ans_grandpa:
                    # 除掉含有已经说过的
                    if parent.identifier not in tree.subtree(a.identifier).nodes:
                        ans_other.append(a)

            if ans_other:
                io_method.out_fun("除此之外，您还需要:")#13
                for a in ans_other:
                    io_method.out_fun(a.tag)#13

        return state_array, state_linklist

    def que_set(self, io_method, s_a_i, i, children_list, state_array, state_linklist):
        if s_a_i == 0:  # ( ==2 or 3) 不该有1的情况了 因为应该已经消歧过了
            io_method.out_fun("您属于 "+children_list[i].tag+" 吗")#14
            x_input = io_method.in_fun()
            if self.check_if_no(x_input):  # 0 不属于
                s_a_i = 3
                state_array[children_list[i].identifier] = 3
                # state_linklist暂时不添加3
            else:
                s_a_i = 2
                state_array[children_list[i].identifier] = 2
                state_linklist.append(children_list[i].identifier)
        return s_a_i

    def answer_for_stuff(self, io_method, tree, nid, state_array, state_linklist):
        ans_node = []
        children_list = tree.children(nid)

        for i in range(len(children_list)):
            s_a_i = state_array[children_list[i].identifier]

            if children_list[i].tag in self.mutex_set:
                flag = 0
                for nid in self.mutex_set_id:
                    # 说明不是当前"军队"的互斥点为2 即不是军队
                    if nid != children_list[i].identifier and state_array[nid] == 2:
                        flag = 1
                        break
                if flag != 1:
                    # 其他互斥点不确定 才反问
                    s_a_i = self.que_set(io_method, s_a_i, i, children_list, state_array, state_linklist)
                    if s_a_i == 2:
                        for n in tree.children(children_list[i].identifier):
                            ans_node.append(n)
            elif children_list[i].tag in self.if_set:
                s_a_i = self.que_set(io_method,s_a_i, i, children_list,state_array, state_linklist)
                if s_a_i == 2:
                    for n in tree.children(children_list[i].identifier):
                        ans_node.append(n)
            elif children_list[i].tag in self.if_two_set:
                s_a_i = self.que_set(io_method, s_a_i, i, children_list,state_array, state_linklist)
                # 租赁 不管有没有租赁 此节点都加上
                ans_node.append(tree.parent(children_list[i].identifier))
                if s_a_i == 2:
                    for n in tree.children(children_list[i].identifier):
                        ans_node.append(n)
            else:
                # 普通材料，流程
                ans_node.append(children_list[i])

        return ans_node, state_array, state_linklist

    def answer_generator_for_include(self, io_method, tree, nid, state_array, state_linklist):
        if not tree.get_node(nid).tag in self.user_set:
            # 材料类
            ans_node, state_array, state_linklist = self.answer_for_stuff(io_method, tree, nid, state_array, state_linklist)
            # 叶节点
            if not ans_node:
                io_method.out_fun("抱歉，超出我的知识范围啦～") #15
                return

            io_method.out_fun(tree.get_node(nid).tag+"包括：") #16
            for n in ans_node:
                io_method.out_fun(n.tag)

        else:
            # 用户类, 不会出现问到叶节点的情况
            state_array, state_linklist = self.answer_for_user(io_method, tree, nid, state_array, state_linklist)

        return state_array, state_linklist

    def generate_answer(self, io_method, id, curr_node=None, children_list=None):
        io_method.io_method.out_fun(self.nlg_answer_array[id])


