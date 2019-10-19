# coding=utf-8
# ! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score,hamming_loss, label_ranking_average_precision_score
from sklearn.model_selection import KFold
from pycm import *

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

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 38, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 2, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("multilabel_threshold", 0.5, "multilabel_threshold (default: 0.5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")

    # for bandian
    df_bandian_pos = pd.read_csv("../resources/database/bandian_q_pos_copy.csv")
    df_two_class = pd.read_csv("../resources/database/two_class_1500.csv")
    df = pd.concat([df_bandian_pos, df_bandian_pos, df_bandian_pos,df_two_class])


    # get data
    x_text, y, vocab_size, max_document_length,word_embedding, sr_word2id = data_helpers.load_data_and_labels(df)
    x_text = np.array(x_text)
    y = np.array(y)

    # Build vocabulary
    # max_document_length = max([len(x.split(" ")) for x in x_text])
    # vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    # x = np.array(list(vocab_processor.fit_transform(x_text)))
    # vocab_processor.save("../resources/database/processor.vocab")
    # print("vocab_processor has saved...")
    # print(len(vocab_processor.vocabulary_))


    # only neg data is random, so below
    # x, x_dev, y, y_dev = train_test_split(x, y, test_size=0, random_state=42)

    # return x, y,vocab_processor
    return x_text, y, vocab_size, max_document_length, word_embedding, sr_word2id


def train(x_train, y_train, vocab_size, x_dev, y_dev, word_embedding):
    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                # vocab_size=len(vocab_processor.vocabulary_),
                vocab_size=vocab_size,
                embedding_size=FLAGS.embedding_dim,
                word_embedding = word_embedding,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs_bandian", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            # vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, sig_scores= sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.sig_scores],
                    feed_dict)
                one_hot_scores = data_helpers.to_one_hot_scores(sig_scores, FLAGS.multilabel_threshold)

                subset_acc = accuracy_score(one_hot_scores, np.array(y_batch))
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, subset_acc {:g}".format(time_str, step, loss, subset_acc))
                train_summary_writer.add_summary(summaries, step)

                return subset_acc


            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, sig_scores= sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.sig_scores],
                    feed_dict)
                one_hot_scores = data_helpers.to_one_hot_scores(sig_scores, FLAGS.multilabel_threshold)
                subset_acc = accuracy_score(one_hot_scores, y_batch)
                ham_loss = hamming_loss(one_hot_scores, y_batch)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, ham {:g}, sub_acc {:g}".format(time_str, step, loss, ham_loss, subset_acc))
                if writer:
                    writer.add_summary(summaries, step)

                return subset_acc

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            pic_train_acc = []
            pic_dev_acc = []
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                pic_train_acc.append(str(train_step(x_batch, y_batch)))
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    pic_dev_acc.append(str(dev_step(x_dev, y_dev, writer=dev_summary_writer)))
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

            with open('pic.txt', 'w') as f:
                f.writelines(" ".join(pic_train_acc))
                f.writelines("\n")
                f.writelines("\n")
                f.writelines(" ".join(pic_dev_acc))

            return cnn, sess, timestamp


def test(cnn, sess, x_test, y_test):

    feed_dict = {
        # may not work x_batch; do work cause (None,1)
        cnn.input_x: x_test,
        cnn.input_y: y_test,
        cnn.dropout_keep_prob: 1.0
    }
    test_loss, test_sig_scores = sess.run([cnn.loss, cnn.sig_scores],feed_dict)
    one_hot_scores = data_helpers.to_one_hot_scores(test_sig_scores, FLAGS.multilabel_threshold)
    subset_acc = accuracy_score(one_hot_scores, y_test)
    ham_loss = hamming_loss(one_hot_scores, y_test)
    rank_pre = label_ranking_average_precision_score(test_sig_scores, y_test)
    print(test_sig_scores)

    print("ham {:g}, sub_acc {:g}, rank_pre {:g}".format(ham_loss, subset_acc, rank_pre))

    # print('val_loss:%f, val_acc:%f' % (test_loss, test_acc))
    #
    # pre = precision_score(test_label, test_pred, average='macro')
    # recall = recall_score(test_label, test_pred, average='macro')
    # acc = accuracy_score(test_label, test_pred)
    # f1 = f1_score(test_label, test_pred, average='macro')


    # cm = ConfusionMatrix(actual_vector=test_label, predict_vector=test_pred)
    # cm.save_obj("../resources/cm_bandian")


def test_unit(x_train, y_train, vocab_size, max_len,word_embedding, sr_word2id):
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    cnn = TextCNN(
        sequence_length=x_train.shape[1],
        num_classes=y_train.shape[1],
        vocab_size=vocab_size,
        embedding_size=FLAGS.embedding_dim,
        word_embedding=word_embedding,
        filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
        num_filters=FLAGS.num_filters,
        l2_reg_lambda=FLAGS.l2_reg_lambda)

    # for fsm_v2
    print("sequence_length=x_train.shape[1]", x_train.shape[1])
    print("num_classes=y_train.shape[1]", y_train.shape[1])

    model_path = "runs_bandian/1554965249/checkpoints/"
    model_file = tf.train.latest_checkpoint(model_path)
    saver = tf.train.Saver()
    saver.restore(sess, model_file)

    # for more
    df_domain_test = pd.read_csv("../resources/test_data/test_classification.csv")
    x_domian = data_helpers.load_data_and_labels_for_test(df_domain_test, max_len,sr_word2id)
    x_domian = np.array(x_domian)


    feed_dict = {
        cnn.input_x: x_domian,
        cnn.dropout_keep_prob: 1.0
    }
    test_sig_scores, scores= sess.run([cnn.sig_scores, cnn.scores], feed_dict)
    # test_sig_scores = test_sig_scores[0]
    # scores = scores
    print(test_sig_scores)
    print(scores)
    print()
    one_hot_scores = data_helpers.to_one_hot_scores(test_sig_scores, FLAGS.multilabel_threshold)
    print("ppp")

    with open('../resources/test_data/test_result.csv', 'wt') as f:
        for i in range(len(df_domain_test)):
            f.write(df_domain_test["user_q"][i])
            f.write("\t")
            f.write(str(df_domain_test["class_label"][i]))
            f.write("\t")
            print(one_hot_scores[i])
            print(test_sig_scores[i])
            print(scores[i])
            print()
            # s = [j for j in range(len(one_hot_scores[i])) if one_hot_scores[i][j] == 1]
            s = []
            if 1 not in one_hot_scores[i]:
                if scores[i][0] > scores[i][1]:
                    max1, max2 = scores[i][0], scores[i][1]
                    max1_j, max2_j = 0, 1
                else:
                    max1, max2 = scores[i][1], scores[i][0]
                    max1_j, max2_j = 1, 0
                for j in range(2, len(scores[i])):
                    if scores[i][j] > max1:
                        max2, max2_j = max1, max1_j
                        max1, max1_j = scores[i][j], j
                    elif max2 < scores[i][j] <= max1:
                        max2, max2_j = scores[i][j], j

                s.append(max1_j)
                s.append(max2_j)
            else:
                for j in range(len(one_hot_scores[i])):
                    if one_hot_scores[i][j] == 1:
                        s.append(j)

            f.write(str(s))
            f.write("\n")


def cv_test(x, y, vocab_size,word_embedding):
    kf = KFold(n_splits=5)
    metrics = []
    c = 0
    for train_index, test_index in kf.split(x):
        x_train, x_dev = x[train_index], x[test_index]
        y_train, y_dev = y[train_index], y[test_index]
        x_dev, x_test, y_dev, y_test = train_test_split(x_dev, y_dev, test_size=0.5, random_state=42)
        # 那用k-fold再循环10次 和我直接train_test_split十次有什么不一样。。。？额 有 我这边完全十次随机，那边是分成十分，每分都有机会的
        print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
        cnn, sess, timestamp = train(x_train, y_train, vocab_size, x_dev, y_dev, word_embedding)
        metrics.append(test(cnn, sess, x_train, y_train))
        c += 1
    print(metrics)
    metrics = np.array(metrics)
    print(metrics.mean(axis=0))


def one_time_train_test(x, y, vocab_size, word_embedding):
    x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.2, random_state=42)
    x_dev, x_test, y_dev, y_test = train_test_split(x_dev, y_dev, test_size=0.5, random_state=42)


    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    cnn, sess, timestamp = train(x_train, y_train, vocab_size, x_dev, y_dev, word_embedding)

    test(cnn, sess, x_test, y_test)


def main(argv=None):
    x, y, vocab_size, max_len,word_embedding, sr_word2id= preprocess()
    # x, y, vocab_processor = preprocess()
    # cv_test(x, y, vocab_processor)
    one_time_train_test(x, y, vocab_size, word_embedding)
    # test_unit(x, y, vocab_size,max_len,word_embedding, sr_word2id)


if __name__ == '__main__':
    tf.app.run()
