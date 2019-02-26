import numpy as np
import tensorflow as tf

from tensorflow.python.ops.nn import dynamic_rnn
from tensorflow.contrib.rnn import GRUCell, LSTMCell, MultiRNNCell
from tensorflow.contrib.lookup.lookup_ops import HashTable, KeyValueTensorInitializer
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import variable_scope

PAD_ID = 0
UNK_ID = 1
GO_ID = 2
EOS_ID = 3
_START_VOCAB = ['_PAD', '_UNK', '_GO', '_EOS']

class EnvModel(object):
    def __init__(self,
            num_symbols,
            num_embed_units,
            num_units,
            num_layers,
            is_train,
            vocab=None,
            embed=None,
            learning_rate=0.1,
            learning_rate_decay_factor=0.95,
            max_gradient_norm=5.0,
            num_samples=512,
            max_length=30,
            use_lstm=True):
        self.epoch = tf.Variable(0, trainable=False, name='epoch')
        self.epoch_add_op = self.epoch.assign(self.epoch + 1)

        self.sessions_input = tf.placeholder(tf.int32, shape=(None, None))   # batch*len
        self.rec_lists = tf.placeholder(tf.int32, shape=(None, None, None))  # batch*len*rec_len
        self.rec_mask = tf.placeholder(tf.float32, shape=(None, None, None))
        #self.aims = tf.placeholder(tf.float32, shape=(None, None, None))
        #self.sessions_length = tf.placeholder(tf.int32, shape=(None))  # batch
        self.purchase = tf.placeholder(tf.float32, shape=(None, None))

        #self.sessions_input = self.symbol2index.lookup(self.sessions)   # batch*len

        # build the embedding table (index to vector)
        if embed is None:
            # initialize the embedding randomly
            self.embed = tf.get_variable('embed', [num_symbols, num_embed_units], tf.float32, initializer=tf.initializers.truncated_normal(0,1))
        else:
            # initialize the embedding by pre-trained word vectors
            self.embed = tf.get_variable('embed', dtype=tf.float32, initializer=embed)

        self.encoder_input = tf.nn.embedding_lookup(self.embed, self.sessions_input) #batch*len*unit

        if use_lstm:
            cell = MultiRNNCell([LSTMCell(num_units) for _ in range(num_layers)])
        else:
            cell = MultiRNNCell([GRUCell(num_units) for _ in range(num_layers)])

        # rnn encoder   [batch_size, length, num_units]
        encoder_output, self.encoder_state = dynamic_rnn(cell, self.encoder_input, 
                None, dtype=tf.float32, scope="encoder")

        #[batch_size, encoder_len, num_embed_units]
        preference = tf.expand_dims(tf.layers.dense(encoder_output, num_embed_units, name="output")[:, -1, :], 1)

        #[batch_size, encoder_len, rec_len, num_embed_units]
        #self.candidate = tf.reshape(tf.gather_nd(self.embed, tf.expand_dims(self.rec_lists, 3)), [tf.shape(self.rec_lists)[0], tf.shape(self.rec_lists)[1], tf.shape(self.rec_lists)[2], num_embed_units])        
        self.candidate = tf.gather_nd(self.embed, tf.expand_dims(self.rec_lists, 3))

        #[batch_size, encoder_length, rec_len]
        logits = tf.reduce_sum(tf.multiply(tf.expand_dims(preference, 2), self.candidate), 3)
        self.prob = tf.nn.softmax(logits)
        mul_prob = self.prob * self.rec_mask

        tmp_prob_sum = tf.expand_dims(tf.reduce_sum(mul_prob, 2), 2)
        self.norm_prob = mul_prob / (tmp_prob_sum + 1e-12)
        #[batch_size, length, 10]
        _, self.index = tf.nn.top_k(self.norm_prob, k=10)

        batch_size, encoder_length, rec_len = tf.shape(logits)[0], tf.shape(logits)[1], tf.shape(logits)[2]
        self.sample_index = tf.reshape(tf.multinomial(tf.reshape(self.norm_prob, [-1, rec_len]), 1), [batch_size, encoder_length, 1])

        self.aims = tf.one_hot(self.index[:,:,0], tf.shape(self.candidate)[2])
        aim_embed = tf.reduce_sum(tf.expand_dims(self.aims, 3) * self.candidate, 2)

        self.score = tf.placeholder(tf.float32, (None))
        self.score_loss = -tf.reduce_mean(tf.expand_dims(self.score, 1) * tf.log(tf.reduce_sum(self.aims * self.norm_prob, 2)[:, -1] + 1e-12))

        #[batch_size, length]
        self.purchase_logits = tf.reduce_mean(tf.multiply(tf.layers.dense(encoder_output, num_units, name="purchase"), tf.layers.dense(aim_embed, num_units, name="aim")), 2)

        # building graph finished and get all parameters
        self.params = tf.trainable_variables()
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, 
                dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        # calculate the gradient of parameters
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        gradients = tf.gradients(self.score_loss, self.params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, 
                max_gradient_norm)
        self.update = opt.apply_gradients(zip(clipped_gradients, self.params), 
                global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2, 
                max_to_keep=10, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))
