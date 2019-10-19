# coding=utf8

import jieba
import jieba.posseg as pseg
import numpy as np
import xlrd
import csv
import pandas as pd
import random
from functools import reduce

showed_dict = set()
with open("/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/resources/show_dict.txt", "r") as f1:
    for line in f1:
        showed_dict.add(line.strip())
        jieba.add_word(line.strip())


#
# words_count = {}
# with open("nor_key_v1.txt", "r") as f1:
#     for line in f1:
#         for word in jieba.cut(line, cut_all=False):
#             if word in words_count:
#                 words_count[word] += 1
#             else:
#                 words_count[word] = 1


def csv_from_excel():
    wb = xlrd.open_workbook('/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/resources/QQA.xlsx')
    sh = wb.sheet_by_name('Sheet1')
    your_csv_file = open('QQA.csv', 'w')
    wr = csv.writer(your_csv_file, quoting=csv.QUOTE_ALL)

    for rownum in range(sh.nrows):
        wr.writerow(sh.row_values(rownum))

    your_csv_file.close()


def get_score(word, pos):
    has_showed = 0
    noun = 0

    if word in showed_dict:
        has_showed = 1

    if "n" in pos or "N" in pos:
        noun = 1

    if word in words_count:
        fre = float(words_count[word]) / len(words_count)
    else:
        fre = 0.0

    # 这样慢，不过为了完整性先这样
    # score = 0.5 * has_showed + 0.4 * noun + 0.1 * fre
    # score = 0.5 * has_showed + 0.4 * noun
    score = 0.5 * has_showed + 0.4 * noun - 0.1 * fre

    return score


def concat_Q_Q1_txt():
    q_list = []
    q1_list = []
    with open("/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/resources/Q.txt", "r") as f1:
        for line in f1:
            q_list.append(line.strip())
    with open("/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/resources/Q1.txt", "r") as f1:
        for line in f1:
            q1_list.append(line.strip())

    X = pd.DataFrame()
    X["user_q"] = q_list
    X["normal_q"] = q1_list

    X.to_csv("/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/resources/QQ.csv", index=False)


def concant_QQ_key():
    keys_set = set()
    with open("/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/resources/keys.txt", "r") as f1:
        for line in f1:
            keys_set.add(line.strip())

    df = pd.read_csv("/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/resources/QQ.csv")
    key_list = []
    for q in df["normal_q"]:
        flag = False
        for key in keys_set:
            if key in q:
                key_list.append(key)
                flag = True
                break
        if flag == False:
            print(q)
            print(" ".join(jieba.cut(q, cut_all=False)))
            key_list.append(q)
            print("not found....")

    assert len(key_list) == len(df)

    df["key"] = key_list
    df.to_csv("/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/resources/QQ.csv", index=False)


def how_many_user_q_not_in_key():
    keys_set = set()
    with open("/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/resources/keys.txt", "r") as f1:
        for line in f1:
            keys_set.add(line.strip())
    df = pd.read_csv("/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/resources/QQ.csv")
    df = df[0:5000]

    count = 0
    for i in range(len(df["user_q"])):
        # print(df["user_q"][i])
        # print(" ".join(jieba.cut(df["user_q"][i], cut_all=False)))
        # print(df["key"][i])
        # a = input()
        flag = False
        for w in jieba.cut(df["user_q"][i], cut_all=False):
            # print("w", w)
            # print("key", df["key"][i])
            # a=input()
            if w == df["key"][i]:
                flag = True
                break
        if flag == False:
            # print(df["key"][i])
            # print(df["user_q"][i])
            # print(" ".join(jieba.cut(df["user_q"][i], cut_all=False)))
            # a=input()
            count += 1
    print("count:", count)


def make_some_positive_samsung_data():
    # 暂时想要30000左右 所以3800+个类 平均每个类要7-8个扩展问就行了 抽不出来了 全部都要
    negs_std = []
    negs_user = []
    negs_class = []
    ele = pd.read_csv("/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/resources/QQ.csv")
    df = pd.DataFrame()
    std_dict = {}
    for s_q, u_q in zip(ele["std_q"], ele["user_q"]):
        if s_q in std_dict:
            std_dict[s_q].append(u_q)
        else:
            std_dict[s_q] = [u_q]

    print(len(std_dict)) # 共有多少个类别

    # 做三星的类别表
    # std_class = []
    # std_id = []
    # num = 0
    # for k, v in std_dict.items():
    #     std_class.append(k)
    #     std_id.append(num)
    #     num += 1
    #
    # X = pd.DataFrame()
    # X["class_id"] = std_id
    # X["class_name"] = std_class
    # X.to_csv("/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/resources/database/samsung_class.csv", index=False)


    # 这种不要了 为了要全部的正
    # 每一个类别 选几个
    # 得到一个候选列表 can_list
    # df_sam_class = pd.read_csv("/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/resources/database/samsung_class.csv")
    # sam_class_dict = {}
    # for id, name in zip(df_sam_class["class_id"],df_sam_class["class_name"]):
    #     sam_class_dict[name] = id
    #
    # # 选的哪几个
    # for s_q in std_dict:
    #     can_list = std_dict[s_q]
    #     num = random.randint(7, 8)
    #     if num <= len(can_list):
    #         for i in range(num):
    #             which = random.randint(0, len(can_list) - 1)
    #             negs_user.append(can_list[which])
    #             negs_std.append(s_q)
    #             negs_class.append(sam_class_dict[s_q])
    #     else:
    #         for q in can_list:
    #             negs_user.append(q)
    #             negs_std.append(s_q)
    #             negs_class.append(sam_class_dict[s_q])
    #
    # df["std_q"] = negs_std
    # df["user_q"] = negs_user
    # df["match_label"] = [1] * len(negs_std)
    # df["class_label"] = negs_class

    #要这个

    df_sam_class = pd.read_csv("/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/resources/database/samsung_class.csv")
    sam_class_dict = {}
    for id, name in zip(df_sam_class["class_id"],df_sam_class["class_name"]):
        sam_class_dict[name] = id

    df["std_q"] = ele["std_q"]
    df["user_q"] = ele["user_q"]
    df["match_label"] = [1] * len(ele)

    for s_q in df["std_q"]:
        negs_class.append(sam_class_dict[s_q])
    df["class_label"] = negs_class

    df.to_csv("/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/resources/samsung_pos.csv", index=False)

def make_some_negative_samsung_data():
    # 想要70000 3894类每个17-18个

    negs_std = []
    negs_user = []
    negs_class_label = []
    ele = pd.read_csv("resources/samsung_pos.csv")

    df_ele = pd.DataFrame()
    std_dict = {}
    user_q_class_label_dict = {}
    for s_q, u_q, c_l in zip(ele["std_q"], ele["user_q"], ele["class_label"]):
        if s_q in std_dict:
            std_dict[s_q].append(u_q)  # same class id
        else:
            std_dict[s_q] = [u_q]

        if u_q not in user_q_class_label_dict:  # one u_q wont belongs to 2 classes
            user_q_class_label_dict[u_q] = c_l
        else:
            print(u_q)  # user_q 还真有重复的，没问题

    print("user_q_class_label_dict", len(user_q_class_label_dict))
    print("std_dict", len(std_dict))
    print(std_dict)

    # 每一个类别 选几个
    # 得到一个候选列表 can_list

    # 还是以user_q的类为主
    for s_q in std_dict:
        can_list = get_can_list(s_q, std_dict)
        for i in range(random.randint(17,18)):
            which = random.randint(0, len(can_list) - 1)
            negs_user.append(can_list[which])
            negs_std.append(s_q)
            negs_class_label.append(user_q_class_label_dict[can_list[which]]) # 这里没有判断是否在字典里，因为都应该在字典里

    df_ele["std_q"] = negs_std
    df_ele["user_q"] = negs_user
    df_ele["match_label"] = [0] * len(negs_std)
    df_ele["class_label"] = negs_class_label
    df_ele.to_csv("resources/samsung_neg.csv", index=False)


def get_can_list(s_q, std_dict):
    can_list = []
    for q in std_dict:
        if not q == s_q:
            for u_q in std_dict[q]:
                can_list.append(u_q)
    return can_list


def make_some_neg_bandian_data():
    negs_std = []
    negs_user = []
    negs_class_label = []
    ele = pd.read_csv("/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/resources/database/bandian_q_pos.csv")

    # X = ele
    # X.to_csv("resources/samsung_pos.csv", index=False)

    df_ele = pd.DataFrame()
    std_dict = {}
    user_q_class_label_dict = {}
    for s_q, u_q, c_l in zip(ele["std_q"], ele["user_q"], ele["class_label"]):
        if s_q in std_dict:
            std_dict[s_q].append(u_q)  # same class id
        else:
            std_dict[s_q] = [u_q]

        if u_q not in user_q_class_label_dict:  # one u_q wont belongs to 2 classes
            user_q_class_label_dict[u_q] = c_l
        else:
            print(u_q)  # user_q 还真有重复的，没问题

    print("user_q_class_label_dict", len(user_q_class_label_dict))
    print("std_dict", len(std_dict))
    print(std_dict)

    # 每一个类别 选几个
    # 得到一个候选列表 can_list

    # 选的哪几个
    for s_q in std_dict:
        can_list = get_can_list(s_q, std_dict)
        for i in range(random.randint(19, 20)):
            which = random.randint(0, len(can_list) - 1)
            negs_user.append(can_list[which])
            negs_std.append(s_q)
            negs_class_label.append(user_q_class_label_dict[can_list[which]]) # 这里没有判断是否在字典里，因为都应该在字典里

    df_ele["std_q"] = negs_std
    df_ele["user_q"] = negs_user
    df_ele["match_label"] = [0] * len(negs_std)
    df_ele["class_label"] = negs_class_label

    df_ele.to_csv("/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/resources/database/bandian_q_neg.csv", index=False)


def change_order_with_node_question():
    df = pd.read_csv("resources/node_question.csv")
    print(df.head())
    X = pd.DataFrame()
    X["std_q"] = df["std_q"]
    X["user_q"] = df["user_q"]
    X["node_name"] = df["node_name"]
    X["node_id"] = df["node_id"]

    X.to_csv("resources/node_question_pos.csv", index=False)


def change_id():
    df = pd.read_csv("resources/database/bandian_entity.csv")
    i_l = []
    for i, id in enumerate(df["name"]):
        i_l.append(i + 1)

    df["id"] = i_l

    df.to_csv("resources/database/bandian_entity.csv", index=False)

def check_class():
    df = pd.read_csv("resources/database/bandian_entity.csv")
    cla = {}
    num = 1
    cl = []
    for n in df["name"]:
        if n in cla:
            cl.append(cla[n])
        else:
            cl.append(num)
            cla[n] = num
        num += 1
    print(len(cla))  # 65
    a=input()

    name = []
    class_id = []
    for k, v in cla.items():
        for i in range(10):
            name.append(k)
            class_id.append(v)

    X = pd.DataFrame()
    X["class_id"] = class_id
    X["class_name"] = name

    X.to_csv("resources/database/bandian_q_pos.csv", index=False)

    # df.to_csv("resources/database/bandian_entity.csv", index=False)

def check_bandian_q_pos():
    df = pd.read_csv("resources/database/node_bandian_q_pos.csv")
    X = pd.DataFrame()
    X["std_q"] = df["std_q"]
    X["user_q"] = df["user_q"]
    X["match_label"] = [1]*len(df)
    X["class_label"] = df["class_id"]  # class_id != id
    # s = set([])
    # for n in df["class_name"]:
    #     s.add(n)
    # l = []
    # for i in s:
    #     l.append(i)
    #
    # Y = pd.DataFrame()
    # Y["cl"] = l


    # num = 1
    # l_n = [1]
    # for i in range(1,len(df)):
    #     if df["std_q"][i] != df["std_q"][i-1]:
    #         num += 1
    #     l_n.append(num)
    #
    # df["class_id"] = l_n
    X.to_csv("resources/database/bandian_q_pos.csv", index=False)
    # df.to_csv("resources/database/node_bandian_q_pos.csv", index=False)
    # Y.to_csv("resources/database/fenci.csv", index=False)


def analyse_matrix():
    df = pd.read_csv("resources/database/classification_test_res.csv")
    class_dict = {}
    tp = []
    for p, e in zip(df["pred"],df["entity"]):
        if p == e:
            tp.append(p)

        if p in class_dict:
            class_dict[p] += 1
        else:
            class_dict[p] = 1

        if e in class_dict:
            class_dict[e] += 1
        else:
            class_dict[e] = 1

    print("num_class:", len(class_dict))  # 44

    tp_dict = {}

    # precision
    pre_dict = {}
    for c in class_dict:
        tp_v = len([x for x in tp if x == c])
        p_v = len([x for x in df["pred"] if x == c])
        if tp_v == 0:
            pre_dict[c] = 0
        else:
            pre_dict[c] = tp_v / p_v

    pre_sum = 0
    for k, v in pre_dict.items():
        pre_sum += v
    print("pre_macro:", pre_sum/len(pre_dict)) # 符合函数算出来的 没问题
    print(pre_dict)

def lists_combination(lists, code=','):

    def myfunc(list1, list2):
        return [str(i) + code + str(j) for i in list1 for j in list2]

    return reduce(myfunc, lists)

def make_2_class_data():
    df = pd.read_csv("/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/resources/database/bandian_q_pos_copy.csv")
    # print(len(list(df["user_q"][9:]))) #748

    # len(set(df["user_q"][9:]))  #734


    class_dict = {}
    flag = 0

    sent_dict = {}
    for i in range(9, len(df)):
        if df["class_label"][i] in class_dict:
            if not flag:
                class_dict[df["class_label"][i]].append(df["user_q"][i])
                flag = 1
                sent_dict[df["user_q"][i]] = df["class_label"][i]

        else:
            flag = 0
            class_dict[df["class_label"][i]] = [df["user_q"][i]]
            sent_dict[df["user_q"][i]] = df["class_label"][i]


    print(class_dict)
    print(len(sent_dict))
    print(sent_dict)

    labels = []

    cla = list(class_dict.values())

    sents = []
    labels = []
    for i, x in enumerate(cla):
        for j in range(i + 1, len(cla)):
            for xi in x:
                for xj in cla[j]:
                    if xi in sent_dict and xj in sent_dict:
                        # print(sent_dict[xi])
                        sents.append(xi + "," + xj)
                        s = str(sent_dict[xi])+","+str(sent_dict[xj])
                        # print(s)
                        labels.append(s)
                    else:
                        print("error")
    print(len(sents))
    print(len(labels))

    X = pd.DataFrame()
    X["query"] = sents
    X["label"] = labels

    X.to_csv("/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/resources/database/two_class.csv", index=True)

def reduce_2_class():
    df = pd.read_csv("/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/resources/database/two_class.csv")
    index = [i for i in range(0, len(df))]
    # print(index)
    index = random.sample(index, len(df)-2000)
    df = df.drop(index)
    df.to_csv("/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/resources/database/two_class_2000.csv", index=False)

def make_2_class_1500():
    df = pd.read_csv("/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/resources/database/two_class_2000.csv")

    df = df.sample(n=1500)
    # 双引号读出来时候好像会没有了
    # for i in range(len(df)):
    #     print(df["query"][i], df["label"][i])
    #     df["query"][i] = df["query"][i].replace('"',"")
    #     df["label"][i] = df["label"][i].replace('"',"")

    X = pd.DataFrame()

    X["user_q"] = df["query"]
    X["std_q"] = [1]*len(df)
    X["match_label"] = [1]*len(df)
    X["class_label"] = df["label"]
    X.to_csv("/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/resources/database/two_class_1500.csv", index=False)


if __name__ == '__main__':
    make_2_class_1500()
    # analyse_matrix()
    # reduce_2_class()
    # make_2_class_data()

    # check_class()
    # check_bandian_q_pos()
    # make_some_neg_bandian_data()
    # make_some_positive_samsung_data()
    # make_some_negative_samsung_data()

    # keylist = []
    # with open("nor_key_v1.txt","r") as f1:
    #     for line in f1:
    #         words = pseg.cut(line.strip())
    #         word_list = []
    #         score_list = []
    #         for word, flag in words:
    #             word_list.append(word)
    #             score_list.append(get_score(word, flag))
    #         # print (np.argmax(score_list))
    #         # a=input()
    #         print(word_list)
    #         print(score_list)
    #         keylist.append(word_list[np.argmax(score_list)])
    #         keylist.append("\n")
    #
    # with open("nor_key_v1_coded.txt", "w") as f2:
    #     f2.writelines(keylist)
    #
    # # 最后再去重一下
