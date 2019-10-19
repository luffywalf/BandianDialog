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
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import KFold
from pycm import *

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 15, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 50, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 3, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


def preprocess():
    # Data Preparation
    # ==================================================
    df_bandian_pos = pd.read_csv("../resources/database/bandian_q_pos.csv")
    df_bandian_neg = pd.read_csv("../resources/database/bandian_q_neg.csv")
    df_bandian = pd.concat([df_bandian_pos, df_bandian_neg])
    x_text, y = data_helpers.load_data_and_labels_bandian(df_bandian)

    # Load data
    print("Loading data...")

    df_intent = pd.read_csv("data/intent_classification_data.csv")
    x_text_intent, y_intent = data_helpers.load_data_and_labels(df_intent)

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    vocab_processor.fit(x_text)
    vocab_processor.save("data/processor.vocab")
    print("vocab_processor has saved...")
    print(len(vocab_processor.vocabulary_))
    a=input()
    x_intent = np.array(list(vocab_processor.transform(x_text_intent)))

    # only neg data is random, so below
    x, x_dev, y, y_dev = train_test_split(x_intent, y_intent, test_size=0, random_state=42)

    return x, y,vocab_processor
    # return x, y, vocab_size, max_len


def train(x_train, y_train, vocab_processor, x_dev, y_dev):
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
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
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
            vocab_processor.save(os.path.join(out_dir, "vocab"))

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
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

            return cnn, sess, timestamp


def test(cnn, sess, x_train, y_train, x_test, y_test, vocab_processor):

    feed_dict = {
        # may not work x_batch; do work cause (None,1)
        cnn.input_x: x_test,
        cnn.input_y: y_test,
        cnn.dropout_keep_prob: 1.0
    }
    test_loss, test_acc, test_pred, test_label = sess.run([cnn.loss, cnn.accuracy, cnn.predictions, cnn.label],
                                                          feed_dict)

    # print(test_p2)
    # test_p2 = [1 for x in test_p2 if x]
    # print("three acc:", sum(test_p2)/len(test_p2))
    print('val_loss:%f, val_acc:%f' % (test_loss, test_acc))

    pre = precision_score(test_label, test_pred, average='macro')
    recall = recall_score(test_label, test_pred, average='macro')
    acc = accuracy_score(test_label, test_pred)
    f1 = f1_score(test_label, test_pred, average='macro')

    # cm = ConfusionMatrix(actual_vector=test_label, predict_vector=test_pred)
    # cm.save_obj("../resources/cm_bandian")

    return [pre, recall, acc, f1]


def test_unit(x_train, y_train, vocab_processor):
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    cnn = TextCNN(
        sequence_length=x_train.shape[1],
        num_classes=y_train.shape[1],
        vocab_size=len(vocab_processor.vocabulary_),
        embedding_size=FLAGS.embedding_dim,
        filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
        num_filters=FLAGS.num_filters,
        l2_reg_lambda=FLAGS.l2_reg_lambda)
    # # for fsm_v2
    print("sequence_length=x_train.shape[1]", x_train.shape[1])
    print("num_classes=y_train.shape[1]", y_train.shape[1])
    print(len(vocab_processor.vocabulary_))
    print(FLAGS.embedding_dim)
    print(list(map(int, FLAGS.filter_sizes.split(","))))
    print(FLAGS.num_filters)
    print(FLAGS.l2_reg_lambda)


    # 这个model是 test 1 acc的 比较好
    model_path = "/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/cnn-intent-classification/runs_bandian/1544259188/checkpoints"
    model_file = tf.train.latest_checkpoint(model_path)
    saver = tf.train.Saver()
    saver.restore(sess, model_file)

    print("1:")
    x_test = data_helpers.sent_to_input("我需要带什么材料吗", vocab_processor)

    feed_dict = {
        cnn.input_x: x_test,
        cnn.dropout_keep_prob: 1.0
    }
    test_pred = sess.run([cnn.predictions], feed_dict)
    pred_class = test_pred[0][0]

    print("class:",pred_class)


    print("2：")
    x_test = data_helpers.sent_to_input("委托授权书呢？", vocab_processor)

    feed_dict = {
        cnn.input_x: x_test,
        cnn.dropout_keep_prob: 1.0
    }
    test_pred = sess.run([cnn.predictions], feed_dict)
    pred_class = test_pred[0][0]

    print("class:", pred_class)

    print("3：")
    x_test = data_helpers.sent_to_input("怎么申请受理", vocab_processor)

    feed_dict = {
        cnn.input_x: x_test,
        cnn.dropout_keep_prob: 1.0
    }
    test_pred = sess.run([cnn.predictions], feed_dict)
    pred_class = test_pred[0][0]

    print("class:", pred_class)


def cv_test(x, y, vocab_processor):
    kf = KFold(n_splits=5)
    metrics = []
    c = 0
    for train_index, test_index in kf.split(x):
        x_train, x_dev = x[train_index], x[test_index]
        y_train, y_dev = y[train_index], y[test_index]
        x_dev, x_test, y_dev, y_test = train_test_split(x_dev, y_dev, test_size=0.5, random_state=42)
        # 那用k-fold再循环10次 和我直接train_test_split十次有什么不一样。。。？额 有 我这边完全十次随机，那边是分成十分，每分都有机会的
        print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
        cnn, sess, timestamp = train(x_train, y_train, vocab_processor, x_dev, y_dev)
        metrics.append(test(cnn, sess, x_train, y_train, x_test, y_test, vocab_processor))
        c += 1
    print(metrics)
    metrics = np.array(metrics)
    print(metrics.mean(axis=0))


def one_time_train_test(x, y, vocab_processor):
    x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.2, random_state=42)
    x_dev, x_test, y_dev, y_test = train_test_split(x_dev, y_dev, test_size=0.5, random_state=42)

    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    cnn, sess, timestamp = train(x_train, y_train, vocab_processor, x_dev, y_dev)

    print(test(cnn, sess, x_train, y_train, x_test, y_test, vocab_processor))

    # print("my own test data")
    # # use_my_own_test_data
    # df_test = pd.read_csv("../resources/database/make_up_test_data.csv")
    # x_test, _ = data_helpers.load_data_and_labels(df_test)
    # x_test = np.array(list(vocab_processor.fit_transform(x_test)))
    # y = df_test["class_label"]
    # y = np.array(y)
    #
    # y_test = np.zeros((len(y), 60))
    # y_test[np.arange(len(y)), y] = 1
    #
    #
    # feed_dict = {
    #     # may not work x_batch; do work cause (None,1)
    #     cnn.input_x: x_test,
    #     cnn.input_y: y_test,
    #     cnn.dropout_keep_prob: 1.0
    # }
    # test_scores, test_pred, test_p3 = sess.run([cnn.scores, cnn.predictions, cnn.predictions3],feed_dict)
    # # test_pred = sess.run([cnn.predictions], feed_dict)
    # print("test_scores")
    # for i in test_scores:
    #     print(i)
    # print("p:", test_pred)
    # test_p3 = test_p3.indices
    #
    # # test_pred = test_pred[0]
    # # test_pred += 1
    #
    # print("test_pred:", test_pred)
    # print("y", y)
    #
    # pre = precision_score(y, test_pred, average='macro')
    # recall = recall_score(y, test_pred, average='macro')
    # acc = accuracy_score(y, test_pred)
    # f1 = f1_score(y, test_pred, average='macro')
    #
    # print("pre, recall, acc, f1:", pre, recall, acc, f1)
    #
    # p3 = []
    # for i, p in enumerate(test_p3):
    #     if y[i] in (p):
    #         p3.append(1)
    #     else:
    #         p3.append(0)
    # print(p3)
    #
    # print("three acc:", sum(p3) / len(p3))
    #
    # cm = ConfusionMatrix(actual_vector=y, predict_vector=test_pred)
    # cm.save_obj("../resources/cm_bandian_own_test")
    #
    # print(test_p3)


def main(argv=None):
    x, y, vocab_processor = preprocess()

    # cv_test(x, y, vocab_processor)
    # one_time_train_test(x, y, vocab_processor)

    test_unit(x, y, vocab_processor)


if __name__ == '__main__':
    tf.app.run()
