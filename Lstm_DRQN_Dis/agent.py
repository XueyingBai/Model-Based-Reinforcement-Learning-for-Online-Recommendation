import numpy as np
import tensorflow as tf

from tensorflow.python.ops.nn import dynamic_rnn
#from tensorflow.contrib.rnn.python.ops.core_rnn_cell import GRUCell, LSTMCell, MultiRNNCell
from tensorflow.contrib.rnn import GRUCell, LSTMCell, MultiRNNCell
#from tensorflow.contrib.seq2seq.python.ops import attention_decoder_fn 
#from tensorflow.contrib.seq2seq.python.ops.seq2seq import dynamic_rnn_decoder
#from tensorflow.contrib.seq2seq.python.ops.loss import sequence_loss
from tensorflow.contrib.lookup.lookup_ops import HashTable, KeyValueTensorInitializer
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import variable_scope

PAD_ID = 0
UNK_ID = 1
GO_ID = 2
EOS_ID = 3
_START_VOCAB = ['_PAD', '_UNK', '_GO', '_EOS']

class AgentModel(object):
    def __init__(self,
            num_symbols,
            num_embed_units,
            num_units,
            num_layers,
            is_train,
            embed=None,
            learning_rate=0.005,
            action_num=10,
            learning_rate_decay_factor=0.95,
            max_gradient_norm=5.0,
            num_samples=512,
            max_length=30,
            use_lstm=True):

        self.sessions_input = tf.placeholder(tf.int32, shape=(None, None))
        #self.sessions_length = tf.placeholder(tf.int32, shape=(None))
        #self.state = tf.placeholder(tf.float32, shape=(None, num_units))    
        self.reward = tf.placeholder(tf.float32, shape=(None))
        self.action = tf.placeholder(tf.int32, shape=(None, None))
        self.action_mask = tf.placeholder(tf.float32, shape=(None, None))
        self.q_target = tf.placeholder(tf.float32, shape=(None))

        # build the embedding table (index to vector)
        if embed is None:
            # initialize the embedding randomly
            self.embed = tf.get_variable('embed', [num_symbols, num_embed_units], tf.float32, initializer=tf.initializers.truncated_normal(0,1))
        else:
            # initialize the embedding by pre-trained word vectors
            self.embed = tf.get_variable('embed', dtype=tf.float32, initializer=embed)

        self.encoder_input = tf.nn.embedding_lookup(self.embed, self.sessions_input) #batch*len*unit
        #self.encoder_input = tf.placeholder(tf.float32, shape=(None, None, num_embed_units))

        if use_lstm:
            cell = MultiRNNCell([LSTMCell(num_units) for _ in range(num_layers)])
        else:
            cell = MultiRNNCell([GRUCell(num_units) for _ in range(num_layers)])
    
        self.lstm_state = tf.placeholder(tf.float32, shape=(2, 2, None, num_units))
        self.ini_state = (tf.contrib.rnn.LSTMStateTuple(self.lstm_state[0,0,:,:], self.lstm_state[0,1,:,:]), tf.contrib.rnn.LSTMStateTuple(self.lstm_state[1,0,:,:], self.lstm_state[1,1,:,:]))

        # rnn encoder   [batch_size, length, num_units]
        self.encoder_output, self.encoder_state = dynamic_rnn(cell, self.encoder_input, 
                None, initial_state=self.ini_state, dtype=tf.float32, scope="encoder")
        self.state = tf.reshape(self.encoder_output, [-1, num_units])
        #[batch_size, num_symbols]
        self.rec_logits = tf.layers.dense(self.state, num_symbols, name="recommendation")
        self.rec_prob = tf.nn.softmax(self.rec_logits)
        #[batch_size]
        self.action_value = tf.reduce_sum(tf.multiply(tf.reduce_sum(tf.one_hot(self.action, num_symbols) * tf.expand_dims(self.action_mask, 2), 1), self.rec_logits), 1)
        # [batch_size, action_num]
        self.sep_value, self.index = tf.nn.top_k(self.rec_logits[:,3:], action_num)
        self.index = self.index + 3
        # [batch_size]
        #self.max_value = tf.reduce_sum(self.sep_value, 1)

        #[batch_size, action_num, num_embed_units]
        self.candidate = tf.gather_nd(self.embed, tf.expand_dims(self.index, 2))

        self.loss = tf.reduce_mean(tf.square((self.q_target - self.action_value)))

        # initialize the training process
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, 
                dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * learning_rate_decay_factor)

        self.epoch = tf.Variable(0, trainable=False, name='epoch')
        self.epoch_add_op = self.epoch.assign(self.epoch + 1)

        self.global_step = tf.Variable(0, trainable=False)
#        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
#        self.train_op = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.loss)        
        self.params = tf.trainable_variables()
        # calculate the gradient of parameters
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        gradients = tf.gradients(self.loss, self.params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, 
                max_gradient_norm)
        self.update = opt.apply_gradients(zip(clipped_gradients, self.params), 
                global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2, 
                max_to_keep=10, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))