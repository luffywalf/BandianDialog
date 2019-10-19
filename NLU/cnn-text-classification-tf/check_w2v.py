import numpy as np
import pandas as pd
# labels_dense=np.asarray([0,3,2,3,1,1])
# num_labels = labels_dense.shape[0]
# print(num_labels)
# num_classes=4
# index_offset = np.arange(num_labels) * num_classes
# print(index_offset)
# labels_one_hot = np.zeros((num_labels, num_classes))
# print(labels_one_hot)
#
# for a, l in zip(labels_one_hot, labels_dense):
#     a[l] = 1
# print(labels_one_hot)


# print(labels_dense.ravel())
# print(index_offset + labels_dense.ravel())
#
# labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
# print(labels_one_hot)

from sklearn.metrics import hamming_loss

# df = pd.read_csv("/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/resources/database/two_class_1500.csv")
# num_labels = len(df)
# num_classes= 60
# labels_one_hot = np.zeros((num_labels, num_classes))
# for a, l in zip(labels_one_hot, df["class_label"]):
#     l = l.split(",")
#     if len(l) == 2:
#         a[int(l[0])], a[int(l[1])] = 1, 1
#     else:
#         a[int(l)] = 1

# print(hamming_loss(np.array([[0, 1], [1, 1],[1,1],[1,1],[1,1]]), np.zeros((5,2))))

x = (2400+1500)*0.8
y = (2400+1500)*0.1
print(x, y)


# from gensim.models import KeyedVectors
# import time
# t1 = time.time()
# print("start to load w2v model...")
# model = KeyedVectors.load_word2vec_format("sgns.merge.word")
# t2 = time.time()
# print("w2v model loaded...")
# print("time used:", t2-t1)
# print(model.similarity('我', '你'))
# print(model.similarity('时间', '明天'))
# print(model.similarity('喜欢', '讨厌'))
# print(model.similarity('手机', '明天'))
# print(model.similarity('习近平', '主席'))
# print(model.similarity('习近平', '国家主席'))