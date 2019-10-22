import NLU
import NLG
import KG
import DM

kg = KG.KG()
tree = kg.build_tree()
tree.show()
nlu = NLU.NLU()
nlg = NLG.NLG()
dm = DM.DM()

# please init first below
# disambiguation_node_dict = {'流程': [4, 19, 43], '材料': [6, 21, 45]}
mutex_dict = {"低压":["高压"], "高压":["低压"], "居民":["非居民"], "非居民":["居民"]}
jumin_id = [3, 18] # 居民 非居民id
dianya_id = [42, 2]  # 高压 低压 id
# mutex_dict = {"低压":["高压","厉害"], "高压":"低压"}
# {'流程': [3, 18, 43], '申请受理': [4, 19, 44], '材料': [5, 20, 45], '有效营业执照复印件': [24, 47], '军队': [29, 52, 65]}

if_type_set = {"高压","居民","非居民"}  # 条件类 统一要判断的
JuMin_ID= 3
FeiJuMin_ID = 18

class FSM:
    def __init__(self):
        self.terminal = 0

    def check_if_type(self, io_method, nid, state_mdp, state_array, state_linklist):
        '''
        如果linklist里面有if_type的 那就看nid在不在对应的if_type_node下 不在就no 在就向后走；
        若没有if_type的，若三者都没有 那只可能是对着低压了（因为有basic_question） 因此只要问居民/非居民
        '''
        for n in state_linklist[::-1]:
            if n != nid and tree.get_node(n).tag in if_type_set:
                # 因为if_type中的元素是互斥的 因此只要进下面一次就可以return了
                if nid in tree.subtree(n).nodes:
                    # 继续向后走
                    state_mdp[0] = 1
                    return 0, state_mdp,state_array,state_linklist
                else:  # 不在就no
                    state_mdp[0] = 2
                    return 1, state_mdp,state_array,state_linklist

        # 没有if_type的 进行反问
        io_method.out_fun("请问您属于 居民还是非居民：")
        rhetoric_list = [tree.get_node(id) for id in jumin_id]
        user_ans_id = nlu.get_rhetoric_ans(io_method, rhetoric_list)

        while user_ans_id == -1:
            io_method.out_fun("请回答上面列出的选项: ")
            user_ans_id = nlu.get_rhetoric_ans(io_method, rhetoric_list)

        # if nlg.check_if_no(x_input):
        #     ID = FeiJuMin_ID
        # else:
        #     ID = JuMin_ID
        ID = user_ans_id

        state_array[ID] = 2
        state_linklist.append(ID)

        if nid in tree.subtree(ID).nodes:
            # 继续向后走
            state_mdp[0] = 1
            return 0, state_mdp,state_array,state_linklist
        else:  # 不在就no
            state_mdp[0] = 2
            return 1, state_mdp,state_array,state_linklist

    def basic_rhetoric(self, io_method, rhetoric_id):
        rhetoric_list = [tree.get_node(id) for id in rhetoric_id]
        user_ans_id = nlu.get_rhetoric_ans(io_method, rhetoric_list)
        while user_ans_id == -1:
            io_method.out_fun("请回答上面列出的选项: ")
            user_ans_id = nlu.get_rhetoric_ans(io_method, rhetoric_list)
        return user_ans_id

    def basic_question(self, io_method, intent_array, state_array, state_linklist):
        # 加上这部分对v2也不需要改什么～
        io_method.out_fun("您好，请问您属于低压还是高压")
        user_ans_id = self.basic_rhetoric(io_method, dianya_id)
        ids = [user_ans_id]
        assert len(ids) == 1

        intent_array[ids[0]] = 0  # if_class
        state_array[ids[0]] = 2
        state_linklist.append(ids[0])

        if tree.get_node(ids[0]).tag == "低压":
            io_method.out_fun("您好，请问您属于居民还是非居民")
            user_ans_id = self.basic_rhetoric(io_method, jumin_id)
            ids = [user_ans_id]
            assert len(ids) == 1

            intent_array[ids[0]] = 0  # if_class
            state_array[ids[0]] = 2
            state_linklist.append(ids[0])

        return intent_array, state_array, state_linklist

    def check_go_back_word(self, mapped_class_name, if_class, ids, intent_array, state_array, state_linklist,disambiguation_node_dict):
        # 假设互斥点后面的第一个歧义点 是要改的歧义点
        if mapped_class_name not in mutex_dict:
            tmp_set = set()
        else:
            tmp_linklist = [tree.get_node(s).tag for s in state_linklist]
            tmp_set = set(mutex_dict[mapped_class_name]) & set(tmp_linklist)
        if len(tmp_set):
            # 这里的tmp_set len = 1; 是互斥点本身state_linklist里面的互斥点本身
            # state_array 1.找要第一个遇到的 要改信息的歧义点
            state_array = [0] * tree.size()  # 因为不好放后面
            mutex_flag = 0
            #这里是为了找互斥点后的第一个歧义点，要是互斥点后没有歧义点 那就没什么变化，外面有接这种情况的
            for s in state_linklist:
                if tree.get_node(s).tag in tmp_set:
                    mutex_flag = 1
                    continue
                if mutex_flag == 1:
                    tag = tree.get_node(s).tag
                    # if not disambiguation_node_dict:
                    if tag in disambiguation_node_dict:
                        # 在互斥点后的第一个歧义点
                        if_class = intent_array[s]  # 现在的设置 若没有歧义点 就dm直接返回了；若有，就取此歧义点对应的intent给后面要消歧的点
                        # new_ids = [] # 这次看 new_ids 后面也没用啊？？有啥存在的必要？
                        for id in disambiguation_node_dict[tag]:
                            state_array[id] = 1
                            # new_ids.append(id)
                        break
            # link_list  互斥点后面的节点信息都不要了
            new_link_list = []
            for s in state_linklist:
                if tree.get_node(s).tag in mutex_dict[mapped_class_name]:
                    break
                new_link_list.append(s)
            # 暂不考虑 互斥点也是歧义点的情况，因此互斥点的ids默认为len = 1
            assert len(ids) == 1
            new_link_list.append(ids[0])
            # # 不知道这里会不会有错：都传回去了 暂时没事
            state_linklist = [x for x in new_link_list]

            # 根据state_linklist改state_array，因为前面把state_array置0了
            for s in state_linklist:
                if state_array[s] != 1:
                    state_array[s] = 2

            # 修改状态后，重新赋值给ids
            ids = [i for i, s in enumerate(state_array) if s == 1]


        return if_class, ids, state_array, state_linklist

    def state_array_2_state_mdp(self, io_method, if_class, state_array,state_linklist, nid):
        state_mdp = [0,0,0]
        if self.terminal:
            state_mdp[2] = 1
        one = [i for i in range(len(state_array)) if state_array[i] == 1]  # 这句话这样写太耗时了 TODO
        if if_class:
            state_mdp[1] = 3
            if len(one) > 1:
                state_mdp[0] = 4
                return state_mdp, state_array,state_linklist
            # 接下来 先统一做（高／居民／非居民）判断 再去判断父节点
            f, state_mdp, state_array,state_linklist = \
                self.check_if_type(io_method, nid, state_mdp, state_array,state_linklist)
            if f:
                return state_mdp,state_array,state_linklist

            if tree.get_node(nid).tag in nlg.if_class_two:
                # 需要两个条件的来判断父节点了
                parent = tree.parent(nid)
                if state_array[parent.identifier] != 2:
                    state_mdp[0] = 3
                else:
                    state_mdp[0] = 1

        else:
            state_mdp[0] = 5
            if len(one) > 1:
                state_mdp[1] = 1
            else:
                state_mdp[1] = 2
        return state_mdp,state_array,state_linklist

    def dm_nlg_step(self, io_method, mapped_class_name, ids, intent_array, state_array, state_linklist, if_class, disambiguation_node_dict):
        print("1:", ids)
        # 反悔信息
        if_class, ids, state_array, state_linklist = self.check_go_back_word(mapped_class_name, if_class, ids,
                                                                    intent_array, state_array,state_linklist, disambiguation_node_dict)
        print("2:", ids)
        # 当反悔但没有歧义点时 走这里；这里就不涉及到intent_array
        ids_len_0_flag = 0
        if len(ids) == 0:
            for i, s in enumerate(state_array):
                if s == 2:
                    entity_node_id_now = i
                    ids_len_0_flag = 1
            io_method.out_fun("好的，您还有什么问题")
        if ids_len_0_flag:
            return ids, intent_array, state_array, state_linklist

        # change state of state_array
        entity_node_id_now = 0  # 如果不是有歧义 这里不会被用上
        for id in ids:
            # 歧义点
            if len(ids) == 1:
                entity_node_id_now = id

            else:
                state_array[id] = 1

        ### state_array -> mdp_state
        mdp_state,state_array,state_linklist = self.state_array_2_state_mdp(io_method, if_class, state_array, state_linklist, entity_node_id_now)
        action_num = dm.get_action(mdp_state)

        # {"greeting": 0, "end": 1, "up_yes": 2, "up_no": 3, "up_q": 4, "down_disa": 5, "down_ans": 6}
        if action_num == 0:
            io_method.out_fun("你好")
        elif action_num == 1:
            self.break_flag = 1
            return ids, intent_array, state_array, state_linklist
        elif action_num == 2:
            io_method.out_fun("那么您需要带～")
            return ids, intent_array, state_array, state_linklist
        elif action_num == 3:
            io_method.out_fun("不需要")
            return ids, intent_array, state_array, state_linklist
        elif action_num == 4:
            parent = tree.parent(entity_node_id_now)
            io_method.out_fun("请问您是否属于"+parent.tag)
            x_input = io_method.in_fun()
            if not nlg.check_if_no(x_input):
                intent_array[parent.identifier] = if_class
                state_array[parent.identifier] = 2
                state_linklist.append(parent.identifier)
                io_method.out_fun("您需要带")
            else:
                io_method.out_fun("不需要")
            return ids, intent_array, state_array, state_linklist
        elif action_num == 5:
            # 假设相同的mapped_class_name对应的ids是相同的；其实
            if mapped_class_name not in disambiguation_node_dict:
                disambiguation_node_dict[mapped_class_name] = ids
            # 可以应对 1.最正常的消歧，2.利用历史信息的消歧， 3.反悔历史的消歧

            entity_node_id_now = nlu.disambiguation(io_method, ids, state_array,state_linklist)
            # 这里是为了将原歧义点中被消去的点的状态改为2 哇
            for id in ids:
                if id != entity_node_id_now:
                    state_array[id] = 0
            print("entity_node_id_now:", entity_node_id_now)
            # 接下来改ids是因为换成MDP结构的话 这里要重新循环上去 不经过input的那种
            ids = [entity_node_id_now]

            intent_array[entity_node_id_now] = if_class
            state_array[entity_node_id_now] = 2
            state_linklist.append(entity_node_id_now)

            ids, intent_array, state_array, state_linklist = self.dm_nlg_step(io_method, mapped_class_name, ids,
                                        intent_array, state_array, state_linklist, if_class,disambiguation_node_dict)

        else:
            nlg.answer_generator_for_include(io_method, tree, entity_node_id_now, state_array, state_linklist)

        return ids, intent_array, state_array, state_linklist

    def process(self, io_method, intent_array, state_array, state_linklist, disambiguation_node_dict, epoch_length=20):
        self.basic_question(io_method, intent_array, state_array, state_linklist)

        self.break_flag = 0
        self.epoch_length = epoch_length
        io_method.out_fun("您有什么问题")
        while self.epoch_length:

            input_x = io_method.in_fun()
            # 结束语
            # add end_set greeting_set
            # end_set = set()
            # greeting_set = set()
            # if input_x in
            if input_x == "再见":
                #  这里其实没必要写成terminal吧。。。这样直接break多少 多省力气 TODO
                self.terminal = 1
                break

            mapped_class_names, if_class = nlu.map(input_x)
            if len(mapped_class_names) == 2:
                name1, name2 = mapped_class_names[0], mapped_class_names[1]
                if (name1 in nlg.user_set and name2 in nlg.user_set) or (name1 not in nlg.user_set and name2 not in nlg.user_set):
                    ids = nlu.map_class_to_entity(tree, name1)
                    ids, intent_array, state_array, state_linklist = self.dm_nlg_step(io_method, name1, ids, intent_array,
                                                                state_array, state_linklist, if_class,disambiguation_node_dict)
                    # 再正常走name2
                    mapped_class_name = name2
                else:
                    # 双实体中有一个属于用户属性, 只有用户属性有可能产生歧义？若有就去消歧，若没有不需要走完整流程
                    # 直接记下来就可以；另一个放到mapped_class_name
                    if name1 in nlg.user_set:
                        name = name1
                        mapped_class_name = name2
                    else:
                        name = name2
                        mapped_class_name = name1
                    ids = nlu.map_class_to_entity(tree, name)
                    if len(ids) > 1:
                        entity_node_id_now = nlu.disambiguation(ids, state_array, state_linklist)
                        for id in ids:
                            if id != entity_node_id_now:
                                state_array[id] = 0
                        print("entity_node_id_now:", entity_node_id_now)
                        intent_array[entity_node_id_now] = if_class
                        state_array[entity_node_id_now] = 2
                        state_linklist.append(entity_node_id_now)
                    else:
                        intent_array[ids[0]] = if_class
                        state_array[ids[0]] = 2
                        state_linklist.append(ids[0])
            else:
                # 单实体
                mapped_class_name = mapped_class_names[0]

            ids = nlu.map_class_to_entity(tree, mapped_class_name)

            # UNK
            if len(ids) == 1 and ids[0] == 0:
                io_method.out_fun("您所说的我不太懂 请换个问题吧")
                continue

            ids, intent_array, state_array, state_linklist = self.dm_nlg_step(io_method, mapped_class_name, ids,
                                            intent_array, state_array, state_linklist, if_class,disambiguation_node_dict)
            if self.break_flag:
                break

            self.epoch_length -= 1



# if __name__ == '__main__':
    # state_array, intent_array = [0] * tree.size(), [0] * tree.size()
    # state_linklist, disambiguation_node_dict = [], {}

    # fsm = FSM()
    # fsm.process(client, intent_array, state_array, state_linklist, disambiguation_node_dict, 20)
