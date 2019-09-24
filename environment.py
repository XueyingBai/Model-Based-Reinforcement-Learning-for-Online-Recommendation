import numpy as np
import tensorflow as tf

from tensorflow.python.ops.nn import dynamic_rnn
from tensorflow.contrib.rnn import GRUCell, LSTMCell, MultiRNNCell
from tensorflow.contrib.lookup.lookup_ops import HashTable, KeyValueTensorInitializer
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import variable_scope
from utils import gen_batched_data, compute_acc
from utils import FLAGS, PAD_ID, UNK_ID, GO_ID, EOS_ID, _START_VOCAB

class EnvModel(object):
    def __init__(self,
            num_symbols,
            num_embed_units,
            num_units,
            num_layers,
            is_train,
            vocab=None,
            embed=None,
            learning_rate=5e-4,
            learning_rate_decay_factor=0.95,
            max_gradient_norm=5.0,
            num_samples=512,
            max_length=30,
            use_lstm=True):
        self.epoch = tf.Variable(0, trainable=False, name='epoch')
        self.epoch_add_op = self.epoch.assign(self.epoch + 1)

        self.sessions_input = tf.placeholder(tf.int32, shape=(None, None))  # batch*len
        self.rec_lists = tf.placeholder(tf.int32, shape=(None, None, None))  # batch*len*rec_len
        self.rec_mask = tf.placeholder(tf.float32, shape=(None, None, None))
        self.aims_idx = tf.placeholder(tf.int32, shape=(None, None))
        self.sessions_length = tf.placeholder(tf.int32, shape=(None))  # batch
        self.purchase = tf.placeholder(tf.float32, shape=(None, None))

        #self.index2symbol = HashTable(KeyValueTensorInitializer(tf.Variable(np.array([i for i in range(num_symbols)], dtype=np.int64), False), 
        #    self.symbols), default_value='_UNK', name="index2symbol")

        batch_size = tf.shape(self.sessions_input)[0]
        encoder_len = tf.shape(self.sessions_input)[1]
        rec_len = tf.shape(self.rec_lists)[2]

        encoder_mask = tf.reshape(tf.cumsum(tf.one_hot(self.sessions_length - 2, 
            encoder_len), reverse=True, axis=1), [-1, encoder_len])

        #self.decoder_mask = tf.reshape(tf.cumsum(tf.one_hot(self.sessions_length-2, 
        #    decoder_len), reverse=True, axis=1), [-1, decoder_len])

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
        encoder_output, _ = dynamic_rnn(cell, self.encoder_input, 
                self.sessions_length, dtype=tf.float32, scope="encoder")
        self.aims = tf.one_hot(self.aims_idx, rec_len)

        # training next click given rec_list
        with tf.variable_scope(""):
            #[batch_size, encoder_len, num_embed_units]
            preference = tf.layers.dense(encoder_output, num_embed_units, name="pref_output")
            #[batch_size, encoder_len, rec_len, num_embed_units]
            self.candidate = tf.reshape(
                tf.gather_nd(self.embed, tf.expand_dims(self.rec_lists, 3)), 
                [batch_size, encoder_len, rec_len, num_embed_units])

            #[batch_size, encoder_length, rec_len]
            #logits = tf.reduce_mean(tf.multiply(tf.expand_dims(preference,2), tf.layers.dense(self.candidate, num_units, name="candidate_output")), 3)
            logits = tf.reduce_mean(tf.multiply(tf.expand_dims(preference,2), self.candidate), 3)            
            ini_prob = tf.nn.softmax(logits)
            mul_prob = ini_prob * self.rec_mask

            tmp_prob_sum = tf.expand_dims(tf.reduce_sum(mul_prob, 2), 2)
            self.norm_prob = mul_prob / (tmp_prob_sum + 1e-12)
            #[batch_size, length, 10]
            _, self.argmax_index = tf.nn.top_k(self.norm_prob, k=FLAGS.metric)
            #self.predict_output = self.index2symbol.lookup(tf.cast(index, dtype=tf.int64))
            #self.predict_loss = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.square(self.aims - self.norm_prob) * self.rec_mask, 2) * encoder_mask, 1)) / tf.reduce_sum(encoder_mask)
            self.predict_loss = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(-self.aims * tf.log(self.norm_prob + 1e-12) * self.rec_mask, 2) * encoder_mask, 1)) / tf.reduce_sum(encoder_mask)

            aim_embed = tf.reduce_sum(tf.expand_dims(self.aims, 3) * self.candidate, 2)
            if FLAGS.use_simulated_data:
                self.purchase_logits, self.purchase_loss = tf.constant(0., dtype=tf.float32), tf.constant(0., dtype=tf.float32)
            else:
                #[batch_size, length]
                self.purchase_logits = tf.reduce_mean(tf.multiply(
                    tf.layers.dense(tf.stop_gradient(encoder_output), num_units, name="purchase_layer"), 
                    tf.layers.dense(tf.stop_gradient(aim_embed), num_units, name="purchase_aim")), 2)

                self.purchase_loss = tf.reduce_sum(tf.reduce_sum(
                    tf.square(self.purchase - tf.nn.sigmoid(self.purchase_logits)) * encoder_mask * tf.pow(self.purchase+1, 5.3), 1)) / tf.reduce_sum(encoder_mask)

        # inference next click given rec_list
        with tf.variable_scope(""):
            tf.get_variable_scope().reuse_variables()
            #[batch_size, length, num_embed_units]
            inf_preference = tf.expand_dims(tf.layers.dense(encoder_output[:,-1,:], num_embed_units, name="pref_output"), 1)
            #[batch_size, length, rec_len, num_embed_units]
            self.inf_candidate = tf.reshape(
                tf.gather_nd(self.embed, tf.expand_dims(self.rec_lists, 3)), 
                [batch_size, 1, rec_len, num_embed_units])

            #[batch_size, 1, rec_len]
            #inf_logits = tf.reduce_mean(tf.multiply(tf.expand_dims(inf_preference,2), tf.layers.dense(self.inf_candidate, num_units, name="candidate_output")), 3)
            inf_logits = tf.reduce_mean(tf.multiply(tf.expand_dims(inf_preference,2), self.inf_candidate), 3)
            inf_ini_prob = tf.nn.softmax(inf_logits)
            inf_mul_prob = inf_ini_prob * self.rec_mask

            inf_tmp_prob_sum = tf.expand_dims(tf.reduce_sum(inf_mul_prob, 2), 2)
            self.inf_norm_prob = inf_mul_prob / (inf_tmp_prob_sum + 1e-12)
            #[batch_size, 1, 10]
            _, self.inf_argmax_index = tf.nn.top_k(self.inf_norm_prob, k=FLAGS.metric)
            inf_aim_embed = tf.reduce_sum(tf.cast(tf.reshape(tf.one_hot(self.inf_argmax_index[:,:,0], rec_len), [batch_size,1,rec_len,1]), tf.float32) * self.inf_candidate, 2)

            if FLAGS.use_simulated_data:
                self.inf_purchase_logits = tf.zeros([1,1], dtype=tf.float32)
            else:
                #[batch_size, 1]
                self.inf_purchase_logits = tf.reduce_mean(tf.multiply(
                    tf.layers.dense(tf.stop_gradient(encoder_output), num_units, name="purchase_layer"), 
                    tf.layers.dense(tf.stop_gradient(inf_aim_embed), num_units, name="purchase_aim")), 2)
        self.decoder_loss = self.predict_loss + self.purchase_loss
        # self.decoder_loss = self.purchase_loss
        # self.decoder_loss = self.predict_loss 

        # building graph finished and get all parameters
        self.params = tf.trainable_variables()
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, 
                dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * learning_rate_decay_factor)

        self.global_step = tf.Variable(0, trainable=False)
        opt = tf.train.AdamOptimizer(self.learning_rate)
        #opt = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.loss)        
        #opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.params = tf.trainable_variables()
        # calculate the gradient of parameters
        gradients = tf.gradients(self.decoder_loss, self.params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, 
                max_gradient_norm)
        self.update = opt.apply_gradients(zip(clipped_gradients, self.params), 
                global_step=self.global_step)

        self.score = tf.placeholder(tf.float32, (None, None))
        self.score_loss = -tf.reduce_mean(self.score * (tf.reduce_sum(self.aims*tf.log(self.norm_prob+1e-18), 2) +
                    tf.log(tf.sigmoid(self.purchase_logits) + 1e-12)))

        score_gradients = tf.gradients(self.score_loss, self.params)
        score_clipped_gradients, self.score_gradient_norm = tf.clip_by_global_norm(score_gradients, 
                max_gradient_norm)
        self.score_update = opt.apply_gradients(zip(score_clipped_gradients, self.params), 
                global_step=self.global_step)
        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2, 
                max_to_keep=10, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))

    def step_decoder(self, session, data, forward_only=False):
        input_feed = {self.sessions_input: data['sessions'],
                self.rec_lists: data['rec_lists'],
                self.aims_idx: data['aims'],
                self.rec_mask: data['rec_mask'],
                self.purchase: data['purchase'],
                self.sessions_length: data['sessions_length']}
        if forward_only:
            output_feed = [self.decoder_loss, self.argmax_index, self.predict_loss, self.purchase_loss, self.purchase_logits]
        else:
            output_feed = [self.decoder_loss, self.argmax_index, self.predict_loss, self.purchase_loss, self.purchase_logits, self.gradient_norm, self.update]
        return session.run(output_feed, input_feed)


    def pg_step_decoder(self, session, data, forward_only=False):
        input_feed = {self.sessions_input:data['sessions'],
                self.rec_lists:data['rec_lists'],
                self.aims_idx: data['aims'],
                self.rec_mask: data['rec_mask'],
                self.score: data['dis_reward'],
                self.sessions_length: data['sessions_length']}
        if forward_only:
            output_feed = [self.score_loss, self.norm_prob]
        else:
            output_feed = [self.score_loss, self.norm_prob, self.score_gradient_norm, self.score_update]

        return session.run(output_feed, input_feed)


    def train(self, sess, dataset, is_train=True, ftest_name="env_test_output.txt"):
        st, ed, loss, acc, acc_1, pr_loss, pu_loss, acc_p = 0, 0, [], [], [], [], [], []
        acc_pur, acc_pur_1 = [], []
        tp, tn, fp, fn = [], [], [], []
        pur_num, pur_num_1 = 0., 0.
        all_reward = []
        print "get %s data:len(dataset) is %d " % ("training" if is_train else "testing", len(dataset))
        if not is_train:
            fout = open(ftest_name, "w")
            fout.close()
        #fout = open("./output.txt", "w")
        while ed < len(dataset):
            # print "training %.4f %%...\r" % (float(ed) / len(dataset) * 100),
            st, ed = ed, ed + FLAGS.batch_size if ed + \
                FLAGS.batch_size < len(dataset) else len(dataset)
            batch_data = gen_batched_data(dataset[st:ed])
            outputs = self.step_decoder(sess, batch_data, forward_only=False if is_train else True)
            loss.append(outputs[0])
            predict_index = outputs[1]  #[batch_size, length, 10]
            pr_loss.append(outputs[2])
            pu_loss.append(outputs[3])
            purchase_logits = outputs[4]
            tmp_acc, tmp_acc_1, pur, all_purchase, pur_1, all_purchase_1 = compute_acc(
                batch_data["aims"], predict_index, batch_data["rec_lists"], batch_data["rec_mask"], batch_data["purchase"], ftest_name=ftest_name, output=(not is_train))
            #print pur, all_purchase, pur_1, all_purchase_1
            acc.append(tmp_acc)
            acc_1.append(tmp_acc_1)
            if all_purchase != 0:
                acc_pur.append(pur / all_purchase)
            if all_purchase_1 != 0:
                acc_pur_1.append(pur_1 / all_purchase_1)
            pur_num += all_purchase
            pur_num_1 += all_purchase_1
            '''
            for i, (s, a, r, p, t) in enumerate(zip(batch_data["sessions"], batch_data["aims"], batch_data["rec_lists"], purchase_logits, batch_data["purchase"])):
                for ss, aa, rr, pp, tt in zip(s,a,r,p,t):
                    if ss == 0:
                        break
                    print >> fout, "click:", ss, "aims:", rr[np.argmax(aa)], "purchase:", tt, "logits:", pp, "action:", list(rr)
                print >> fout, "------------next session:%d------------" % i
            '''
            if not FLAGS.use_simulated_data:
                all_num, p_num, true_pos, true_neg, false_pos, false_neg = 1e-6, 0., 0., 0., 0., 0.
                for b_pu, b_pu_l in zip(batch_data["purchase"], purchase_logits):
                    for pu, pu_l in zip(b_pu, b_pu_l): 
                        if pu != -1.:
                            #print pu, pu_l
                            all_num += 1
                            if pu == 1. and pu_l > 0:
                                p_num += 1
                                true_pos += 1
                            if pu == 1. and pu_l <= 0:
                                false_neg += 1
                            if pu == 0. and pu_l > 0:
                                false_pos += 1 
                            if pu == 0. and pu_l <= 0:
                                p_num += 1
                                true_neg += 1
                acc_p.append(p_num / all_num)

                tp.append(true_pos / all_num)
                tn.append(true_neg / all_num)
                fp.append(false_pos / all_num)
                fn.append(false_neg / all_num)
        print "predict:p@1:", str(np.mean(acc_1) * 100)+"%", "p@%d:"%FLAGS.metric, str(np.mean(acc)*100)+"%"
        if not FLAGS.use_simulated_data:
            print "purchase:", str(np.mean(acc_p)*100)+"%"
            print "purchase:"
            print "tp:", np.mean(tp)
            print "tn:", np.mean(tn)
            print "fp:", np.mean(fp) 
            print "fn:", np.mean(fn)
            print "acc in only purchase data:", np.mean(acc_pur), np.mean(acc_pur_1), pur_num, pur_num_1
        if is_train:
            sess.run(self.epoch_add_op)
        return np.mean(loss), np.mean(pr_loss), np.mean(pu_loss), np.mean(acc), np.mean(acc_1)

    def pg_train(self, sess, dataset):
        st, ed, loss = 0, 0, []
        print "get %s data:len(dataset) is %d " % ("training", len(dataset))
        #fout = open("./output.txt", "w")
        while ed < len(dataset):
            #print "epoch %d, training %.4f %%...\r" % (epoch, float(ed) / len(dataset) * 100),
            st, ed = ed, ed + FLAGS.batch_size if ed + \
                FLAGS.batch_size < len(dataset) else len(dataset)
            batch_data = gen_batched_data(dataset[st:ed])
            outputs = self.pg_step_decoder(sess, batch_data, forward_only=False)
            loss.append(outputs[0])
        sess.run(self.epoch_add_op)
        return np.mean(loss)
