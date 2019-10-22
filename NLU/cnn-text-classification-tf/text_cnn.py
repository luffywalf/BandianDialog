import tensorflow as tf
# tf.reset_default_graph()


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, word_embedding ,filter_sizes, num_filters, if_one_entity, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embedding_size = word_embedding.shape[1]
            self.W = tf.get_variable(name='word_embedding', shape=word_embedding.shape, dtype=tf.float32,
                                     initializer=tf.constant_initializer(word_embedding), trainable=False)
            # self.W = tf.Variable(
            #     tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
            #     name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters] # 这个1是rgb channel数？我觉得是
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],  # 后两个 in_channels, out_channels
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        print("self.h_pool:", self.h_pool)
        print("self.h_pool_flat:", self.h_pool_flat)


        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output") as scope:
            W = tf.get_variable(
                scope + "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)  # 这个应该是正则项
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            if not if_one_entity:
                self.sig_scores = tf.sigmoid(self.scores, name="sig_scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.label = tf.argmax(self.input_y, 1, name="label")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            if if_one_entity:
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            else:
                losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            #  tf.argmax(self.input_y, 1) 1 refers to row; tf.equal return true or false;
            #  here name="x" for train.py to find x
            correct_predictions = tf.equal(self.predictions, self.label)
            #  tf.cast to change type; tf.reduce_mean to get mean from all the numbers
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")