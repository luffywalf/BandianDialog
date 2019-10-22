# coding=utf-8
import numpy as np
import re
from io import open
import jieba
import pandas as pd
from gensim.models import KeyedVectors
import pickle as pkl
import time

word2id_pkl_path = "/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/cnn-text-classification-tf/PKL/sgns.chinese/word2id.pkl"
word_emb_path = "/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/cnn-text-classification-tf/PKL/sgns.chinese/word_embedding.npy"


def build_glove_dic():
    # print("1:")
    # model = KeyedVectors.load_word2vec_format("/Users/dingning/Ding/WordEmbedding/vectors.bin")
    # model = KeyedVectors.load_word2vec_format("/home/dn/WordEmbedding/vectors.bin")
    # vocab = model.vocab



    # # print(model.similarity('习近平', '国家主席'))
    # print(model.similarity('我', '你'))
    # print(model.similarity('时间', '明天'))
    # print(model.similarity('喜欢', '讨厌'))
    # print(model.similarity('手机', '明天'))
    # print(model.similarity('习近平', '主席'))
    # '''
    # 1:
    # 0.8819308989411313
    # 0.21165871339936546
    # 0.8029626688859705
    # 0.149779715183571
    # 0.33458088165653804

    # print("2:")

    # 下面如果load如果加了编码参数会有编码问题
    model = KeyedVectors.load_word2vec_format("/home/dn/WordEmbedding/sgns.merge.word")
    vocab = model.vocab
    # print(model.similarity('我', '你'))
    # print(model.similarity('时间', '明天'))
    # print(model.similarity('喜欢', '讨厌'))
    # print(model.similarity('手机', '明天'))
    # print(model.similarity('习近平', '主席'))
    # # print(model.similarity('习近平', '国家主席'))
    # 0.5361739384278194
    # 0.21796330595601968
    # 0.4137033871398726
    # 0.1224886671268881
    # 0.5186771436983313

    # print("3:")
    # model = KeyedVectors.load_word2vec_format("/Users/dingning/Ding/WordEmbedding/newsblogbbs.vec")
    # print(model["是"])

    # vocab = model.vocab
    # 
    # print(model.similarity('我', '你'))
    # print(model.similarity('时间', '明天'))
    # print(model.similarity('喜欢', '讨厌'))
    # print(model.similarity('手机', '明天'))
    # print(model.similarity('习近平', '主席'))
    #
    # a=input()

    # '''
    # 3:
    # 0.7092660606762036
    # 0.18631768554791647
    # 0.7054504772076913
    # -0.06473733357228542
    # -0.22659847855939522
    #
    # '''

    sr_word2id = pd.Series(range(2, len(vocab) + 2), index=vocab)

    sr_word2id['<pad>'] = 0
    sr_word2id['<unk>'] = 1
    word_embedding = []
    for v in vocab:
        word_embedding.append(model[v])

    word_embedding = np.array(word_embedding)
    print("init emb len", len(word_embedding))
    print("word_embedding.shape:", word_embedding.shape)
    word_pad = np.array([0]*(word_embedding.shape)[1])
    word_mean = np.mean(word_embedding, axis=0)  # for unk
    word_mean = np.vstack([word_pad, word_mean])
    word_embedding = np.vstack([word_mean, word_embedding])
    print("final embed len:", len(word_embedding))

    sr_word2id.to_pickle("src/generate/PKL/word2id.pkl")  # python3 no protocol python2 protocol=2
    np.save('src/generate/PKL/word_embedding.npy', word_embedding)
    print("word_embedding loaded...")

    return sr_word2id, word_embedding


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def pad(max_len, q):
    l = len(q)
    if l < max_len:
        for i in range(max_len - l):
            q.append(0)  # should i use other symbol?
    if l > max_len:
        q = q[:max_len]

    assert len(q) == max_len
    return q

def load_data_and_labels_for_test(df, max_document_length,sr_word2id):
    x_text = []
    for q in df["user_q"]:
        x_text.append(" ".join(jieba.cut(q, cut_all=False)))

    print(x_text)

    # 2
    unk_set = set([])
    unk_list = []
    all_words_count = 0
    x_seq = []
    for q in x_text:
        seq = []
        for w in q.split(" "):
            all_words_count += 1
            if w in sr_word2id:
                seq.append(int(sr_word2id[w]))
            else:
                unk_set.add(w)
                unk_list.append(w)
                seq.append(1)
        if len(seq) != max_document_length:
            seq = pad(max_document_length, seq)

        x_seq.append(seq)

    print("unk_list_len", len(unk_list))
    print("unk_set_len", len(unk_set))
    print("total:", all_words_count)

    print("load_data_and_labels_done...")
    return x_seq


def load_data_and_labels_two_entities(df):
    '''
    for 双实体
    :param df: 
    :return: [x_text, y]
    
    # 1.make user_q segmented, y change to one-hot 
    # 2.turn x_text to num sequence and pad sentence
    
    # TODO:  may save the vocab and do speed up
    '''

    print("start load_data_and_labels...")
    t1 = time.time()
    # sr_word2id, word_embedding = build_glove_dic()
    sr_word2id = pd.read_pickle(word2id_pkl_path)
    word_embedding = np.load(word_emb_path)
    t2 = time.time()
    print("Vocabulary Size: {:d}".format(len(sr_word2id)))
    print("load emd use time:", t2-t1)
    vocab_size = len(sr_word2id)

    # 1
    num_labels = len(df)
    num_classes = 60
    labels_one_hot = np.zeros((num_labels, num_classes))
    for a, l in zip(labels_one_hot, df["class_label"]):
        # 若label为int
        if type(l) == type(1):
            a[l] = 1
        else:
            # 若label为字符串"2,4"
            l = l.split(",")
            a[int(l[0])], a[int(l[1])] = 1, 1

    y = labels_one_hot
    x_text = []
    for q in df["user_q"]:
        x_text.append(" ".join(jieba.cut(q, cut_all=False)))

    # 2
    unk_set = set([])
    unk_list = []
    all_words_count = 0
    max_document_length = max([len(x.split(" ")) for x in x_text])  # 9：22
    x_seq = []
    for q in x_text:
        seq = []
        for w in q.split(" "):
            all_words_count += 1
            if w in sr_word2id:
                seq.append(int(sr_word2id[w]))
            else:
                unk_set.add(w)
                unk_list.append(w)
                seq.append(1) # unk应该是1
        if len(seq) != max_document_length:
            seq = pad(max_document_length, seq)

        x_seq.append(seq)
    assert len(x_seq) == len(x_text)
    print("unk_list_len",len(unk_list))
    print("unk_set_len", len(unk_set))
    print("total:", all_words_count)

    print("load_data_and_labels_done...")
    return x_seq, y, vocab_size, max_document_length, word_embedding, sr_word2id
    # return x_seq, y, vocab_size, max_document_length


def load_data_and_labels_one_entity(df):
    '''
    for 单实体
    :param df:
    :return: [x_text, y]

    # 1.make user_q segmented, y change to one-hot
    # 2.turn x_text to num sequence and pad sentence
    '''

    print("start load_data_and_labels...")
    t1 = time.time()
    # sr_word2id, word_embedding = build_glove_dic()
    sr_word2id = pd.read_pickle(word2id_pkl_path)
    word_embedding = np.load(word_emb_path)
    t2 = time.time()
    print("Vocabulary Size: {:d}".format(len(sr_word2id)))
    print("load emd use time:", t2 - t1)
    vocab_size = len(sr_word2id)

    # 1
    y = pd.get_dummies(df["class_label"]).values
    x_text = []
    for q in df["user_q"]:
        x_text.append(" ".join(jieba.cut(q, cut_all=False)))

    # 2
    unk_set = set([])
    unk_list = []
    all_words_count = 0
    max_document_length = max([len(x.split(" ")) for x in x_text])  # 9：22
    x_seq = []
    for q in x_text:
        seq = []
        for w in q.split(" "):
            all_words_count += 1
            if w in sr_word2id:
                seq.append(int(sr_word2id[w]))
            else:
                unk_set.add(w)
                unk_list.append(w)
                seq.append(1)  # unk应该是1
        if len(seq) != max_document_length:
            seq = pad(max_document_length, seq)

        x_seq.append(seq)
    assert len(x_seq) == len(x_text)
    print("unk_list_len", len(unk_list))
    print("unk_set_len", len(unk_set))
    print("total:", all_words_count)

    print("load_data_and_labels_done...")
    return x_seq, y, vocab_size, max_document_length, word_embedding, sr_word2id
    # return x_seq, y, vocab_size, max_document_length



def sent_to_input(q, vocab_processor):
    # sr_word2id = pd.read_pickle("PKL/sgns.weibo/word2id.pkl")
    x_text = " ".join(jieba.cut(q, cut_all=False))
    # attention: here below [text] , text for one line test
    x_text = list(vocab_processor.fit_transform([x_text]))
    return x_text

    # attention: here is [text] ,not text for one line test
    # x_seq = []
    # for w in x_text.split(" "):
    #     if w in sr_word2id:
    #         x_seq.append(sr_word2id[w])
    #     else:
    #         x_seq.append(0)
    # if len(x_seq) != max_len:
    #     x_seq = pad(max_len, x_seq)

    # return np.array([x_seq])
    # return np.array([x_text])

def to_one_hot_scores(scores, ts):
    for i, x in enumerate(scores):
        for j, y in enumerate(x):
            if y > ts:
                scores[i][j] = 1
            else:
                scores[i][j] = 0
    # print("one_hot_scores:", scores)
    return scores


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
