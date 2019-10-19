# coding=utf8

import tensorflow as tf
import pandas as pd
from tensorflow.contrib import learn
from text_cnn import TextCNN, TextCNN2
import jieba
import numpy as np
import data_helpers
import time

import KG
import NLG

# dont forget () here next time
nlg = NLG.NLG()
kg = KG.KG()
tree = kg.build_tree()


# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("multilabel_threshold", 0.5, "multilabel_threshold (default: 0.5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


class NLU:
    def __init__(self):
        self.word2id_pkl_path = "/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/cnn-text-classification-tf/PKL/sgns.chinese/word2id.pkl"
        self.word_emb_path = "/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/cnn-text-classification-tf/PKL/sgns.chinese/word_embedding.npy"
        self.text_model_path = "/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/cnn-text-classification-tf/runs_bandian/1554965249/checkpoints/"
        self.intent_vocab_pro = "src/generate/processor.vocab"
        self.intent_model_path = "/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/cnn-intent-classification/runs_bandian/1544259188/checkpoints"
        self.bandian_node = "/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/resources/database/node_bandian_q_pos.csv"

        self.sr_word2id = pd.read_pickle(self.word2id_pkl_path)
        print("start load embedding...")
        t1 = time.time()
        word_embedding = np.load(self.word_emb_path)
        t2 = time.time()
        print("load emd use time:", t2 - t1)

        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        self.sess = tf.Session(config=session_conf)
        self.cnn = TextCNN(
            sequence_length=52,
            num_classes=60,
            vocab_size=1292483,
            embedding_size=FLAGS.embedding_dim,
            word_embedding=word_embedding,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        model_path = self.text_model_path
        model_file = tf.train.latest_checkpoint(model_path)
        saver = tf.train.Saver()
        saver.restore(self.sess, model_file)
        print("load text classification done...")

        ##############
        self.vocab_processor = learn.preprocessing.VocabularyProcessor.restore(self.intent_vocab_pro)

        tf.reset_default_graph()  # output_1 not found error
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        self.sess_intent = tf.Session(config=session_conf)
        self.cnn_intent = TextCNN2(
            sequence_length=29,
            num_classes=2,
            vocab_size=len(self.vocab_processor.vocabulary_), # 前面 lhs rhs mismatch的错误就是因为两边的vocab不一样大
            # vocab_size=294,
            embedding_size=128,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)
        model_path_intent = self.intent_model_path
        model_file_intent = tf.train.latest_checkpoint(model_path_intent)
        saver_intent = tf.train.Saver()
        saver_intent.restore(self.sess_intent, model_file_intent)

        print("load intent classification done...")

    def sent_to_input(self, q, vocab_processor):
        x_text = " ".join(jieba.cut(q, cut_all=False))
        # attention: here below [text] , text for one line test

        x_text = list(vocab_processor.fit_transform([x_text]))
        return x_text

    def map_class_to_entity(self, tree, mapped_class_name):
        # here thanks to my mentor Yu for a solid code style
        if mapped_class_name == "UNK":
            return [0]
        return [x.identifier for x in tree.filter_nodes(lambda x: x.tag == mapped_class_name)]

    def find_disambiguate_start_id(self, ids, state_linklist):
        # 这里非常神奇的如果不传回来,ids就会是lca后的值 我现在知道为什么了 python就是这样的 列表是传地址引用

        # 先用已有的点消除一些没必要的点得到新的ids(may len(new_ids)<len(ids) ), 然后再找new_ids的lca
        # tmp_disambiguate_start_id_list = [lca_id]

        andset_length_of_statelinklist = []
        for s_id in state_linklist:
            and_set = set([x for x in tree.subtree(s_id).nodes]) & set(ids)
            # 有1用1 state_linklist有能直接消歧的元素
            if len(and_set) == 1:
                for e in and_set:
                    id = e
                return id, 1, ids
            andset_length_of_statelinklist.append(len(and_set))
        # 如果andset_length_of_statelinklist加和为0 说明ids与已有的点没有任何交集，那么找消岐点就直接用ids就好了
        new_ids = ids
        if sum(andset_length_of_statelinklist) != 0:
            # andset_length_of_statelinklist不会再有1
            tmp_len = andset_length_of_statelinklist[0]
            tmp_ind = 0

            print("andset_length_of_statelinklist：", andset_length_of_statelinklist)

            # 除了1，0 之外 len最小的那个对应的点最该最为消歧开始点
            for i, l in enumerate(andset_length_of_statelinklist):
                if 1 < l < tmp_len:
                    tmp_ind = i

            new_ids = list(set([x for x in tree.subtree(state_linklist[tmp_ind]).nodes]) & set(ids))

        ids, lca_id = kg.get_latest_common_ancestor(tree, new_ids)


        # 后面的0／1是为了外面的判断，为1可以直接消歧返回
        return lca_id, 0, ids

    def get_rhetoric_ans(self, io_method, children_list, user_ans_id=-1):
        user_ans = io_method.in_fun()
        mapped_class_name, _ = self.map(user_ans)
        for i, child in enumerate(children_list):
            # mapped_class_name type: list
            if child.tag == mapped_class_name[0]:
                user_ans_id = child.identifier
        return user_ans_id

    def disambiguation(self, io_method,ids, state_array, state_linklist):
        ids_tmp = [x for x in ids]
        while 1:
            disa_id, one_flag,ids_tmp = self.find_disambiguate_start_id(ids_tmp, state_linklist)

            # 可以做成 每次不断确认新的find_disambiguate_start_id 现在暂时用不到就没做
            # disa_id_set = set([x for x in ids])

            if one_flag == 1:
                # 这里为了代码统一
                id = disa_id
                return id

            children_list = nlg.rhetoric_generator(io_method,tree, disa_id, ids_tmp)
            user_ans_id = self.get_rhetoric_ans(io_method, children_list)

            while user_ans_id == -1:
                io_method.out_fun("请回答上面列出的选项: ")
                user_ans_id = self.get_rhetoric_ans(io_method, children_list)

            next_id = user_ans_id
            # 这里反问得到的结果应该都是非歧义点 TODO 这里可以加error
            # 这里没加intent_array 基本没问题？
            state_array[next_id] = 2
            state_linklist.append(next_id)

            # TODO 这里的tree.subtree(1).nodes是包括了自己本身哦 ，再想想会不会有出错情况
            and_set = set([x for x in tree.subtree(next_id).nodes]) & set(ids_tmp)

            if len(and_set) == 1:
                break

            ids_tmp = and_set

        # 随便初始化了一下，正常是会被覆盖的 TODO error
        id = -1
        for e in and_set:
            id = e

        return id


    def map(self, input_x, multi_entity=True):
        # args: sequence_length, num_classes, vocab_processor, model_path, input_x
        # model path 在初始化里改

        x_test = data_helpers.load_data_and_labels_for_test([input_x], 52, self.sr_word2id)

        feed_dict = {
            self.cnn.input_x: x_test,
            self.cnn.dropout_keep_prob: 1.0
        }
        pred_class = []
        if multi_entity:
            test_sig_scores, scores = self.sess.run([self.cnn.sig_scores, self.cnn.scores], feed_dict)
            one_hot_scores = data_helpers.to_one_hot_scores(test_sig_scores, FLAGS.multilabel_threshold)
            print(one_hot_scores)

            # 只有一句话
            # 这里跟初始NLU不同 若没有1的情况 就只取其中的max
            if 1 not in one_hot_scores[0]:
                pred_class.append(np.argmax(one_hot_scores[0]))
            else:
                for j in range(len(one_hot_scores[0])):
                    if one_hot_scores[0][j] == 1:
                        pred_class.append(j)

        else:
            x_test = self.sent_to_input(input_x, self.vocab_processor)

            feed_dict = {
                self.cnn.input_x: x_test,
                self.cnn.dropout_keep_prob: 1.0
            }

            test_pred = self.sess.run([self.cnn.predictions], feed_dict)
            pred_class.append(test_pred[0][0])

        ####intent
        x_test = self.sent_to_input(input_x, self.vocab_processor)

        feed_dict = {
            self.cnn_intent.input_x: x_test,
            self.cnn_intent.dropout_keep_prob: 1.0
        }

        test_pred = self.sess_intent.run([self.cnn_intent.predictions], feed_dict)
        pred_class_intent = test_pred[0][0]

        ####intent_end

        # for visual
        df_ele = pd.read_csv(self.bandian_node)

        pred_class_str = []
        for i in range(len(df_ele)):
            if df_ele["class_id"][i] in pred_class:
                if df_ele["class_name"][i] not in pred_class_str:
                    print("ppp")
                    print(df_ele["class_name"][i])
                    print(pred_class_str)
                    pred_class_str.append(df_ele["class_name"][i])

        print("mapped result:", pred_class_str, pred_class,pred_class_intent)
        # 第二个参数 0代表非条件（含义类型）；1代表条件类型
        return pred_class_str, pred_class_intent





if __name__ == '__main__':
    nlu = NLU()
    # input_x = "什么有效身份证明"
    # nlu.map(input_x)
    input_x = "我是机关事业单位的，需不需要报装申请资料"
    nlu.map(input_x)

    #接下来写 KG 首先 流程 - 》 3个流程

# error
# 办电流程是什么样的 机关事业单位
# 主题证明 - 流程 还要重新问 但其实有主体证明已经是高压。因为是向上找能判断的，没有向下找 ---》这应该比较好弄 测定是谁的孩子
#
# NLU部分错误（单个实体）
# 低压 -> 居民 or 非居民 这个可以做下限制
# 军队
# 边的关系不全
# 前端