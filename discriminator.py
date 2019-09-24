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
from utils import FLAGS, gen_batched_data

PAD_ID = 0
UNK_ID = 1
GO_ID = 2
EOS_ID = 3
_START_VOCAB = ['_PAD', '_UNK', '_GO', '_EOS']

class DisModel(object):
    def __init__(self,
            num_symbols,
            num_embed_units,
            num_units,
            num_layers,
            is_train,
            vocab=None,
            embed=None,
            learning_rate=0.01,
            learning_rate_decay_factor=0.95,
            beam_size=5,
            max_gradient_norm=5.0,
            num_samples=512,
            max_length=30,
            use_lstm=True):

        self.sessions_input = tf.placeholder(tf.int32, shape=(None, None))  # batch*len
        self.sessions_length = tf.placeholder(tf.int32, shape=(None))
        self.rec_lists = tf.placeholder(tf.int32, shape=(None, None, None))  # batch*len*rec_len
        self.rec_mask = tf.placeholder(tf.float32, shape=(None, None, None))
        self.aims_idx = tf.placeholder(tf.int32, shape=(None, None))
        self.label = tf.placeholder(tf.int32, shape=(None))
        self.purchase = tf.placeholder(tf.float32, shape=(None, None))

        self.epoch = tf.Variable(0, trainable=False, name='dis/epoch')
        self.epoch_add_op = self.epoch.assign(self.epoch + 1)
        batch_size = tf.shape(self.sessions_input)[0]
        encoder_len = tf.shape(self.sessions_input)[1]
        rec_len = tf.shape(self.rec_lists)[2]

        encoder_mask = tf.reshape(tf.cumsum(tf.one_hot(self.sessions_length - 2, 
            encoder_len), reverse=True, axis=1), [-1, encoder_len])

        # build the embedding table (index to vector)
        if embed is None:
            # initialize the embedding randomly
            self.embed = tf.get_variable('dis/embed', [num_symbols, num_embed_units], tf.float32)
        else:
            # initialize the embedding by pre-trained word vectors
            self.embed = tf.get_variable('dis/embed', dtype=tf.float32, initializer=embed)

        self.encoder_input = tf.nn.embedding_lookup(self.embed, self.sessions_input) #batch*len*unit

        if use_lstm:
            cell = MultiRNNCell([LSTMCell(num_units) for _ in range(num_layers)])
        else:
            cell = MultiRNNCell([GRUCell(num_units) for _ in range(num_layers)])

        # rnn encoder
        encoder_output, _ = dynamic_rnn(cell, self.encoder_input, self.sessions_length, dtype=tf.float32, scope="dis/encoder")

        #[batch_size, len, num_embed_units]
        self.preference = tf.layers.dense(encoder_output, num_units, name="dis/out2preference")
        #[batch_size, len, rec_len, num_units]
        self.candidate = tf.layers.dense(tf.nn.embedding_lookup(self.embed, self.rec_lists), num_units, name="dis/rec2candidate")
        #[batch_size, len, rec_len]
        self.pre_mul_can = tf.reduce_sum(tf.expand_dims(self.preference, 2) * self.candidate, 3)

        #[batch_size, num_units]
        #self.max_output = tf.reduce_sum(tf.reduce_max(self.pre_mul_can * tf.expand_dims(self.rec_mask, 3), 2) * tf.expand_dims((self.purchase*2-1)*encoder_mask, 2), 1) / tf.expand_dims(tf.reduce_sum(encoder_mask, 1), 1)

        #[batch_size, num_units]
        #self.output = tf.reduce_sum(tf.reduce_sum(self.pre_mul_can * tf.expand_dims(self.aims, 3), 2), 1) / tf.expand_dims(tf.reduce_sum(encoder_mask, 1), 1)
        #self.logits = tf.layers.dense(self.output * self.max_output, 2, name="dis")

        # self.max_embed = tf.reduce_sum(tf.expand_dims(tf.one_hot(tf.argmax(self.pre_mul_can, 2), rec_len), 3) * self.candidate, 2)
        self.max_embed = tf.reduce_sum(tf.expand_dims(tf.nn.softmax(self.pre_mul_can / 0.1), 3) * self.candidate, 2)
        self.aim_embed = tf.reduce_sum(tf.expand_dims(tf.one_hot(self.aims_idx, rec_len), 3) * self.candidate, 2)
        if FLAGS.use_simulated_data:
            purchase_weight = 1.0
        else:
            W_p = tf.get_variable("Wp", shape=(), dtype=tf.float32)
            b_p = tf.get_variable("bp", shape=(), dtype=tf.float32)
            purchase_weight = self.purchase * W_p + b_p
        self.logits = tf.reduce_sum(tf.reduce_sum(self.max_embed * self.aim_embed, 2) * purchase_weight * encoder_mask, 1) / tf.reduce_sum(encoder_mask, 1)
        self.prob = tf.nn.sigmoid(self.logits)
        self.decoder_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.cast(self.label, tf.float32)))
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.greater(self.prob, 0.5), tf.int32), self.label), tf.float32))

        # building graph finished and get all parameters
        self.params = tf.trainable_variables()

        # initialize the training process
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False,
                dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * learning_rate_decay_factor)
        self.assign_lr_op = self.learning_rate.assign(0.01)

        self.global_step = tf.Variable(0, trainable=False)

        # calculate the gradient of parameters
        #opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        opt = tf.train.MomentumOptimizer(self.learning_rate, 0.9)

        gradients = tf.gradients(self.decoder_loss, self.params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, 
                max_gradient_norm)
        self.update = opt.apply_gradients(zip(clipped_gradients, self.params), 
                global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2, 
                max_to_keep=10, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))

    def step_decoder(self, session, data, forward_only=False):
        input_feed = {
            self.sessions_input : data["sessions"],
            self.sessions_length : data["sessions_length"],
            self.label : data["labels"],
            self.rec_lists: data['rec_lists'],
            self.rec_mask: data['rec_mask'],  
            self.aims_idx: data['aims'],
            self.purchase: data['purchase']      
        }
        if forward_only:
            output_feed = [self.decoder_loss, self.acc, self.prob]
        else:
            output_feed = [self.decoder_loss, self.acc, self.prob, self.gradient_norm, self.update]
        return session.run(output_feed, input_feed)


    def train(self, data, data_gen, sess, is_train=True):
        st, ed, loss, acc = 0, 0, [], []
        while ed < len(data):
            # print "dis_epoch %d, training %.4f %%...\r" % (self.epoch.eval(session=sess), float(ed) / len(data) * 100),
            st, ed = ed, ed+32 if ed+32 < len(data) else len(data)
            st_gen, ed_gen = st % len(data_gen), ed % len(data_gen)
            tmp_data_gen = data_gen[st_gen:ed_gen] if st_gen < ed_gen else data_gen[st_gen:] + data_gen[:ed_gen]

            concat_data = list(data[st:ed]) + tmp_data_gen
            batch_data = gen_batched_data(concat_data)
            batch_data["labels"] = np.array(list(np.array([1]*(ed-st))) + list(np.array([0]*len(tmp_data_gen))))
            # for key in batch_data:
                # print key, np.shape(batch_data[key])
            # exit()
            if is_train:
                outputs = self.step_decoder(sess, batch_data)
            else:
                outputs = self.step_decoder(sess, batch_data, forward_only=True)
            loss.append(outputs[0])
            acc.append(outputs[1])

        if is_train:
            sess.run(self.epoch_add_op)
        return np.mean(loss), np.mean(acc)
