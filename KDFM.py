import numpy as np
import tensorflow as tf
import config

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from time import time


class DeepAFM(BaseEstimator, TransformerMixin):

    def __init__(self, feature_size_one_hot, field_size_one_hot,  # one-hot feature parameters
                 feature_size_multi_value, field_size_multi_value,  # multi-value feature parameters
                 embedding_size=8, attention_size=10,  # AFM parameters
                 deep_layers=None, dropout_deep=None, deep_layer_activation=tf.nn.relu,  # DNN parameters
                 epoch=10, batch_size=1024, learning_rate=0.001, optimizer="adam",  # training parameters
                 use_afm=True, use_deep=True, random_seed=2018,  # random parameters
                 loss_type="mse", eval_metric=mean_squared_error, l2_reg=0.0,  # evaluating parameters
                 rnn_size=201, num_rnn_layers=1, keep_lstm=0.5, num_unroll_steps=80, field_size_text=3,  # LSTM parameters
                 topics=None, word_embeddings=None, verbose=False  # word vector parameters
                 ):

        self.feature_size = feature_size_one_hot + feature_size_multi_value + embedding_size * field_size_text  # f
        self.feature_size_one_hot = feature_size_one_hot  # fo
        self.feature_size_multi_value = feature_size_multi_value  # fm
        self.field_size = field_size_one_hot + field_size_multi_value + field_size_text * topics  # F
        self.field_size_one_hot = field_size_one_hot  # Fo
        self.field_size_multi_value = field_size_multi_value  # Fm
        self.field_size_text = field_size_text  # Ft

        self.embedding_size = embedding_size  # K
        self.attention_size = attention_size  # A

        self.deep_layers = deep_layers
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layer_activation

        self.epoch = epoch
        self.batch_size = batch_size  # N
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer

        self.use_afm = use_afm
        self.use_deep = use_deep
        self.random_seed = random_seed

        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.l2_reg = l2_reg

        self.train_result, self.valid_result = [], []
        self.mae_train_result, self.mae_valid_result = [], []

        self.rnn_size = rnn_size
        self.num_rnn_layers = num_rnn_layers
        self.dropout_keep_lstm = keep_lstm
        self.num_unroll_steps = num_unroll_steps  # sentence length

        self.topics = topics  # T

        self.word_embeddings = word_embeddings
        self.verbose = verbose

        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():

            tf.set_random_seed(self.random_seed)

            self.label = tf.placeholder(tf.float32, shape=[None, 1], name='label')  # label
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name='dropout_keep_deep')
            self.dropout_keep_lstm = tf.placeholder(tf.float32, shape=None, name='dropout_deep_lstm')

            # one-hot feature part
            if self.field_size_one_hot > 0:
                self.feat_index = tf.placeholder(tf.int32, shape=[None, None], name='feat_index_one_hot')
                self.feat_value = tf.placeholder(tf.float32, shape=[None, None], name='feat_value_one_hot')

            # multi_value feature part
            if self.field_size_multi_value > 0:
                self.feat_index_m = tf.placeholder(tf.int32, shape=[self.field_size_multi_value, None, None], name='feat_index_multi_value')
                self.feat_value_m = tf.placeholder(tf.float32, shape=[self.field_size_multi_value, None, None], name='feat_value_multi_value')

            # text feature part
            if self.field_size_text > 0:
                self.text_data = tf.placeholder(tf.int32, [None, self.field_size_text, self.num_unroll_steps])  # N * Ft * S

            self.weights = self._initialize_weights()  # TODO:权值没有更新

            # Embeddings
            # one-hot feature
            if self.field_size_one_hot > 0:
                self.embeddings_one_hot = tf.nn.embedding_lookup(self.weights['feature_embeddings_one_hot'], self.feat_index)  # N * Fo * K
                feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size_one_hot, 1])
                self.embeddings_one_hot = tf.multiply(self.embeddings_one_hot, feat_value)  # N * Fo * K

            # TODO: 维度有问题
            # multi_value feature
            if self.field_size_multi_value > 0:
                embeddings_multi_value_list = []
                for i in range(self.field_size_multi_value):  # Fm
                    embeddings_multi_value = tf.nn.embedding_lookup(self.weights['feature_embeddings_multi_value'][i], self.feat_index_m[i])  # N * fmi * K
                    embeddings_nonzero = tf.count_nonzero(self.feat_index_m[i], axis=1)  # N * K
                    feat_value_m = tf.reshape(self.feat_value_m[i, :, :], shape=[-1, self.feature_size_multi_value[i], 1])
                    embeddings_multi_value = tf.multiply(embeddings_multi_value, feat_value_m)  # N * fmi * K
                    embeddings_multi_value = tf.reduce_sum(embeddings_multi_value, axis=1)  # N * K
                self.embeddings_multi_value = tf.stack(embeddings_multi_value_list)  # Ft * N * K
                self.embeddings_multi_value = tf.transpose(self.embeddings_text, perm=[1, 0, 2])  # N * Ft * K

            # text feature
            if self.field_size_text > 0:
                self.word_embeddings = tf.Variable(tf.to_float(self.word_embeddings), trainable=True, name="word_embeddings")  # 字典长度 * E
                embeddings_text_list = []
                for i in range(self.field_size_text):
                    embeddings_text_list.append(self.bilstm_network(self.text_data[:, i], i))  # N * (T * K)
                self.embeddings_text = tf.stack(embeddings_text_list)  # Ft * N * (T * K)
                self.embeddings_text = tf.transpose(self.embeddings_text, perm=[1, 0, 2, 3])  # N * Ft * T * K
                self.embeddings_text = tf.reshape(self.embeddings_text, shape=[-1, self.field_size_text * self.topics,
                                                                               self.embedding_size])  # N * (Ft * T) *K

            # concat feature
            if self.field_size_one_hot > 0:
                self.embeddings = self.embeddings_one_hot
                if self.field_size_multi_value > 0:
                    self.embeddings = tf.concat([self.embeddings, self.embeddings_multi_value], axis=1)  # N * F * K
                if self.field_size_text > 0:
                    self.embeddings = tf.concat([self.embeddings, self.embeddings_text], axis=1)  # N * F * K
            elif self.field_size_text > 0:
                self.embeddings = self.embeddings_text

            # AFM component
            # element_wise
            element_wise_product_list = []
            for i in range(self.field_size):
                for j in range(i + 1, self.field_size):
                    element_wise_product_list.append(tf.multiply(self.embeddings[:, i, :], self.embeddings[:, j, :]))  # N * K

            self.element_wise_product = tf.stack(element_wise_product_list)  # [F * (F - 1)/2] * N * K
            self.element_wise_product = tf.transpose(self.element_wise_product, perm=[1, 0, 2], name='element_wise_product')  # N * [F * (F - 1)/2] *  K

            # attention part
            num_interactions = int(self.field_size * (self.field_size - 1) / 2)
            # wx+b -> relu(wx+b) -> h*relu(wx+b)
            self.attention_wx_plus_b = tf.reshape(tf.add(tf.matmul(tf.reshape(self.element_wise_product, shape=(-1, self.embedding_size)), self.weights['attention_w']),
                self.weights['attention_b']),
                shape=[-1, num_interactions, self.attention_size])  # N * [F * (F - 1)/2] * A

            self.attention_exp = tf.exp(tf.reduce_sum(tf.multiply(tf.nn.relu(self.attention_wx_plus_b),
                                                                  self.weights['attention_h']),
                                                      axis=2, keepdims=True))  # N * [F * (F - 1) / 2] * 1

            self.attention_exp_sum = tf.reduce_sum(self.attention_exp, axis=1, keepdims=True)  # N * 1 * 1

            self.attention_out = tf.div(self.attention_exp, self.attention_exp_sum, name='attention_out')  # N * [F * (F - 1)/2] * 1

            self.attention_x_product = tf.reduce_sum(tf.multiply(self.attention_out, self.element_wise_product), axis=1, name='afm')  # N * K

            self.attention_part_sum = tf.matmul(self.attention_x_product, self.weights['attention_p'])  # N * 1

            # first order term
            if self.field_size_one_hot > 0:
                self.y_first_order = tf.nn.embedding_lookup(self.weights['feature_bias'], self.feat_index)
                self.y_first_embedding_one_hot = tf.multiply(self.y_first_order, feat_value)
                self.y_first_embedding = tf.concat([self.y_first_embedding_one_hot, self.embeddings_text], axis=1)
                self.y_first_order = tf.reduce_sum(self.y_first_embedding, 2)

            # bias
            self.y_bias = self.weights['bias'] * tf.ones_like(self.label)  # N * 1

            # out
            self.out_afm = tf.add_n([tf.reduce_sum(self.y_first_order, axis=1, keepdims=True),
                                     self.attention_part_sum,
                                     self.y_bias], name='out_afm')  # N * 1

            # Deep component
            self.y_deep = tf.reshape(self.embeddings, shape=[-1, self.field_size * self.embedding_size])
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])

            for i in range(0, len(self.deep_layers)):
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" % i]), self.weights["bias_%d" % i])
                self.y_deep = self.deep_layers_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[i + 1])  # N * D

            # out
            self.out_deep = tf.add(tf.matmul(self.y_deep, self.weights['deep_projection']), self.weights['deep_bias'])  # N * 1

            # concat output
            if self.use_afm and self.use_deep:
                concat_input = tf.concat([self.out_afm, self.out_deep], axis=1)
                self.out = tf.add(tf.matmul(concat_input, self.weights['concat_projection']), self.weights['concat_bias'])  # N * 1
            elif self.use_afm:
                self.out = self.out_afm
            elif self.use_deep:
                self.out = self.out_deep

            # loss
            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == "mse":
                #self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))
                self.loss = tf.reduce_mean(tf.square(self.label-self.out))

            # l2 regularization on weights
            if self.l2_reg > 0:
                if self.use_afm and self.use_deep:
                    self.loss += tf.contrib.layers.l2_regularizer(
                        self.l2_reg)(self.weights["concat_projection"])
                if self.use_deep:
                    self.loss += tf.contrib.layers.l2_regularizer(
                        self.l2_reg)(self.weights["deep_projection"])
                    for i in range(len(self.deep_layers)):
                        self.loss += tf.contrib.layers.l2_regularizer(
                            self.l2_reg)(self.weights["layer_%d" % i])
                self.loss += tf.add_n(tf.get_collection('losses'))

            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

    def _initialize_weights(self):
        weights = dict()

        # embeddings
        weights['feature_embeddings'] = tf.Variable(
            tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01),
            name='feature_embeddings')
        weights['feature_embeddings_one_hot'] = tf.Variable(
            tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01),
            name='feature_embeddings_one_hot')
        weights['feature_bias'] = tf.Variable(tf.random_normal([self.feature_size, self.embedding_size], 0.0, 1.0),
                                              name='feature_bias')
        weights['bias'] = tf.Variable(tf.constant(0.1), name='bias')

        # attention part
        glorot = np.sqrt(2.0 / (self.attention_size + self.embedding_size))

        weights['attention_w'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.embedding_size, self.attention_size)),
            dtype=tf.float32, name='attention_w')

        weights['attention_b'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.attention_size,)),
                                             dtype=tf.float32, name='attention_b')

        weights['attention_h'] = tf.Variable(np.random.normal(loc=0, scale=1, size=(self.attention_size,)),
                                             dtype=tf.float32, name='attention_h')

        weights['attention_p'] = tf.Variable(np.ones((self.embedding_size, 1)), dtype=np.float32)

        # deep layers
        num_layer = len(self.deep_layers)
        input_size = self.field_size * self.embedding_size
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))

        weights['layer_0'] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32
        )
        weights['bias_0'] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])), dtype=np.float32
        )

        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]
            weights["bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                dtype=np.float32)  # 1 * layer[i]

        weights['deep_projection'] = tf.Variable(np.random.normal(loc=0, scale=glorot,
                                                                  size=(self.deep_layers[-1], 1)), dtype=np.float32)
        weights['deep_bias'] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        # final concat projection layer
        if self.use_deep and self.use_afm:
            glorot = np.sqrt(2.0 / 3)
            weights['concat_projection'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(2, 1)),
                                                       dtype=np.float32)
            weights['concat_bias'] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        return weights

    def get_batch(self, Xi, Xv, Xim, Xvm, Xt, y, batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end], Xv[start:end], Xim[start:end], Xvm[start:end], Xt[start:end], [[y_] for y_ in y[start:end]]

    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self, a, b, c, d, e, f):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)
        np.random.set_state(rng_state)
        np.random.shuffle(d)
        np.random.set_state(rng_state)
        np.random.shuffle(e)
        np.random.set_state(rng_state)
        np.random.shuffle(f)

    def evaluate(self, Xi, Xv, Xim, Xvm, Xt, y):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :param y: label of each sample in the dataset
        :return: metric of the evaluation
        """
        y_pred = self.predict(Xi, Xv, Xim, Xvm, Xt)
        return self.eval_metric(y, y_pred), mean_absolute_error(y, y_pred)

    def predict(self, Xi, Xv, Xim, Xvm, Xt):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :return: predicted probability of each sample
        """
        # dummy y
        dummy_y = [1] * len(Xi)
        batch_index = 0
        Xi_batch, Xv_batch, Xim_batch, Xvm_batch, Xt_batch,y_batch = self.get_batch(Xi, Xv,  Xim, Xvm, Xt, dummy_y,self.batch_size, batch_index)
        y_pred = None
        while len(Xi_batch) > 0:
            num_batch = len(y_batch)
            feed_dict = {self.label: y_batch,
                         self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
                         self.dropout_keep_lstm: 0.5}

            if self.field_size_one_hot > 0:
                feed_dict[self.feat_index] = Xi_batch
                feed_dict[self.feat_value] = Xv_batch
            if self.field_size_multi_value > 0:
                feed_dict[self.feat_index_m] = Xim_batch
                feed_dict[self.feat_value_m] = Xvm_batch
            if self.field_size_text > 0:
                feed_dict[self.text_data] = Xt_batch

            batch_out = self.sess.run(self.out, feed_dict=feed_dict)

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            batch_index += 1
            Xi_batch, Xv_batch, Xim_batch, Xvm_batch, Xt_batch, y_batch = self.get_batch(Xi, Xv, Xim, Xvm, Xt, dummy_y,self.batch_size, batch_index)

        return y_pred

    def fit_on_batch(self, Xi, Xv, Xim, Xvm, Xt, y):
        feed_dict = {self.label: y,
                     self.dropout_keep_deep: self.dropout_deep,
                     self.dropout_keep_lstm: 0.5}
        if self.field_size_one_hot > 0:
            feed_dict[self.feat_index] = Xi
            feed_dict[self.feat_value] = Xv
        if self.field_size_multi_value > 0:
            feed_dict[self.feat_index_m] = Xim
            feed_dict[self.feat_value_m] = Xvm
        if self.field_size_text > 0:
            feed_dict[self.text_data] = Xt

        loss, opt = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)

        return loss

    def fit(self, Xi_train, Xv_train, Xim_train, Xvm_train, Xt_train, y_train,
            Xi_valid=None, Xv_valid=None, Xim_train_valid=None, Xvm_train_valid=None, Xt_train_valid=None, y_valid=None):

        has_valid = Xv_valid is not None
        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train, Xt_train, Xim_train, Xvm_train)
            total_batch = int(len(y_train) / self.batch_size)
            for i in range(total_batch):
                Xi_batch, Xv_batch, Xim_batch, Xvm_batch, Xt_batch, y_batch = self.get_batch(Xi_train, Xv_train, Xim_train, Xvm_train, Xt_train, y_train, self.batch_size, i)
                self.fit_on_batch(Xi_batch, Xv_batch, Xim_batch, Xvm_batch, Xt_batch, y_batch)

            # evaluate training and validation datasets
            train_result, mae_train_result = self.evaluate(Xi_train, Xv_train, Xim_train, Xvm_train, Xt_train, y_train)
            self.train_result.append(train_result)
            self.mae_train_result.append(mae_train_result)

            if has_valid:
                valid_result, mae_valid_result = self.evaluate(Xi_valid, Xv_valid, Xim_train_valid, Xvm_train_valid, Xt_train_valid, y_valid)
                self.valid_result.append(valid_result)
                self.mae_valid_result.append(mae_valid_result)
            if self.verbose > 0 and epoch % self.verbose == 0:
                if has_valid:
                    print("[%d] train-result=%.4f, valid-result=%.4f, mae_train-result=%.4f, mae_valid-result=%.4f [%.1f s]"
                          % (epoch + 1, train_result, valid_result, mae_train_result, mae_valid_result, time() - t1))
                else:
                    print("[%d] train-result=%.4f [%.1f s]"
                          % (epoch + 1, train_result, time() - t1))


    def bilstm_network(self, input_data, number):

        with tf.variable_scope("text_feature_%s" % (str(number))):
            # build BILSTM network
            # forward rnn
            fw_lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.rnn_size)  # R
            fw_cell = tf.nn.rnn_cell.MultiRNNCell([fw_lstm_cell] * self.num_rnn_layers, state_is_tuple=True)  # R
            # backforward rnn
            bw_lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.rnn_size)  # R
            bw_cell = tf.nn.rnn_cell.MultiRNNCell([bw_lstm_cell] * self.num_rnn_layers, state_is_tuple=True)  # R

            # embedding layer
            inputs = tf.nn.embedding_lookup(self.word_embeddings, input_data)  # N * S * E

            inputs = tf.nn.dropout(inputs, self.dropout_keep_lstm)  # N * S * E

            inputs = [tf.squeeze(input, [1]) for input in tf.split(inputs, self.num_unroll_steps, 1)]  # S * N * E

            out_put, _, _ = tf.nn.static_bidirectional_rnn(fw_cell, bw_cell, inputs,
                                                           dtype=tf.float32)  # S * N * (R * 2)
            raw_out_put = tf.transpose(out_put, perm=[1, 0, 2])  # N * S * (R * 2)

            topic_list = []
            for i in range(self.topics):
                with tf.variable_scope("topic_%d" % i):
                    out_put = attention(raw_out_put, self.attention_size, self.l2_reg)  # N * (R * 2)
                    out_put = tf.nn.dropout(out_put, self.dropout_keep_lstm)  # N * (R * 2)

                    out_put = topic(out_put, self.embedding_size, self.l2_reg)
                    out_put = tf.nn.dropout(out_put, self.dropout_keep_lstm)  # N * K

                topic_list.append(out_put)
            out_puts = tf.stack(topic_list)  # T * N * K
            out_puts = tf.transpose(out_puts, perm=[1, 0, 2])  # N * T * K
            #out_puts = tf.reshape(out_puts, shape=[out_puts.get_shape()[0].value, -1])  # N * (T * K)

            return out_puts


def attention(inputs, attention_size, l2_reg):
    """
    Attention mechanism layer.
    :param inputs: outputs of RNN/Bi-RNN layer (not final state)
    :param attention_size: linear size of attention weights
    :return: outputs of the passed RNN/Bi-RNN reduced with attention vector
    """
    # In case of Bi-RNN input we need to concatenate outputs of its forward and backward parts
    #print(1)
    if isinstance(inputs, tuple):
        inputs = tf.concat(2, inputs)

    sequence_length = inputs.get_shape()[1].value  # the length of sequences processed in the antecedent RNN layer
    #print(inputs.get_shape())
    hidden_size = inputs.get_shape()[2].value  # hidden size of the RNN layer

    # Attention mechanism
    W_omega = tf.get_variable("W_omega", initializer=tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.get_variable("b_omega", initializer=tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.get_variable("u_omega", initializer=tf.random_normal([attention_size], stddev=0.1))

    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
    exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

    # Output of Bi-RNN is reduced with attention vector
    output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)

    l2_loss = 0
    if l2_reg > 0:
        l2_loss += tf.contrib.layers.l2_regularizer(l2_reg)(W_omega)
        l2_loss += tf.contrib.layers.l2_regularizer(l2_reg)(b_omega)
        l2_loss += tf.contrib.layers.l2_regularizer(l2_reg)(u_omega)
        tf.add_to_collection('losses', l2_loss)
    return output

def topic(input, embedding_size, l2_reg):

    hidden_size = input.get_shape()[1].value  # hidden size of the RNN layer

    w = tf.get_variable("lstm_embedding_w",
                        initializer=tf.random_normal([hidden_size, embedding_size],
                                                     stddev=0.1))
    b = tf.get_variable("lstm_embedding_b",
                        initializer=tf.random_normal([embedding_size], stddev=0.1))

    out_put = tf.add(tf.matmul(input, w), b)  # N * K
    out_put = tf.nn.relu(out_put)  # N * K

    l2_loss = 0
    if l2_reg > 0:
        l2_loss += tf.contrib.layers.l2_regularizer(l2_reg)(w)
        l2_loss += tf.contrib.layers.l2_regularizer(l2_reg)(b)
        tf.add_to_collection('losses', l2_loss)

    return  out_put

