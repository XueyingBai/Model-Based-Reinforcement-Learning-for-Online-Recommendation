import numpy as np
import tensorflow as tf
import random

from tensorflow.python.ops.nn import dynamic_rnn
#from tensorflow.contrib.rnn.python.ops.core_rnn_cell import GRUCell, LSTMCell, MultiRNNCell
from tensorflow.contrib.rnn import GRUCell, LSTMCell, MultiRNNCell
#from tensorflow.contrib.seq2seq.python.ops import attention_decoder_fn 
#from tensorflow.contrib.seq2seq.python.ops.seq2seq import dynamic_rnn_decoder
#from tensorflow.contrib.seq2seq.python.ops.loss import sequence_loss
from tensorflow.contrib.lookup.lookup_ops import HashTable, KeyValueTensorInitializer
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import variable_scope
from output_projection import output_projection_layer
from utils import gen_batched_data, compute_acc
from utils import FLAGS, PAD_ID, UNK_ID, GO_ID, EOS_ID, _START_VOCAB

class AgentModel(object):
    def __init__(self,
            num_symbols,
            num_embed_units,
            num_units,
            num_layers,
            is_train,
            embed=None,
            learning_rate=1e-4,
            action_num=10,
            learning_rate_decay_factor=0.95,
            max_gradient_norm=5.0,
            num_samples=512,
            max_length=30,
            use_lstm=True):

        self.sessions_input = tf.placeholder(tf.int32, shape=(None, None))
        self.sessions_length = tf.placeholder(tf.int32, shape=(None))
        self.reward = tf.placeholder(tf.float32, shape=(None))
        self.aims_idx = tf.placeholder(tf.int32, shape=(None, None))
        
        self.rec_lists = tf.placeholder(tf.int32, shape=(None, None, None))  # batch*len*rec_len
        self.rec_mask = tf.placeholder(tf.float32, shape=(None, None, None))

        # build the embedding table (index to vector)
        if embed is None:
            # initialize the embedding randomly
            self.embed = tf.get_variable('embed', [num_symbols, num_embed_units], tf.float32, initializer=tf.initializers.truncated_normal(0,1))
        else:
            # initialize the embedding by pre-trained word vectors
            self.embed = tf.get_variable('embed', dtype=tf.float32, initializer=embed)

        batch_size = tf.shape(self.sessions_input)[0]
        encoder_length = tf.shape(self.sessions_input)[1]
        rec_length = tf.shape(self.rec_lists)[2]

        self.mask = tf.reshape(tf.cumsum(tf.one_hot(self.sessions_length - 2, 
            encoder_length), reverse=True, axis=1), [-1, encoder_length])
        #[batch_size, length]
        self.sessions_target = tf.concat([self.sessions_input[:, 1:], tf.ones([batch_size, 1], dtype=tf.int32)*PAD_ID], 1)

        self.encoder_input = tf.nn.embedding_lookup(self.embed, self.sessions_input) #batch*len*unit
        #self.encoder_input = tf.placeholder(tf.float32, shape=(None, None, num_embed_units))

        if use_lstm:
            cell = MultiRNNCell([LSTMCell(num_units) for _ in range(num_layers)])
        else:
            cell = MultiRNNCell([GRUCell(num_units) for _ in range(num_layers)])

        self.aims = tf.one_hot(self.aims_idx, rec_length)

        #training
        with tf.variable_scope(""):
            output_fn, sampled_sequence_loss = output_projection_layer(num_units, num_symbols)
            self.encoder_output, self.encoder_state = dynamic_rnn(cell, self.encoder_input, 
                    self.sessions_length, dtype=tf.float32, scope="encoder")

            tmp_dim_1 = tf.tile(tf.reshape(tf.range(batch_size), [batch_size, 1, 1, 1]), [1, encoder_length, rec_length, 1])
            tmp_dim_2 = tf.tile(tf.reshape(tf.range(encoder_length), [1, encoder_length, 1, 1]), [batch_size, 1, rec_length, 1])
            #[batch_size, el, rl, 3]
            gather_idx = tf.concat([tmp_dim_1, tmp_dim_2, tf.expand_dims(self.rec_lists, 3)], 3)

            #[batch_size*length, num_symbols], [batch_size*length]
            y_prob, local_loss, total_size, y_log_prob = sampled_sequence_loss(self.encoder_output, self.sessions_target, self.mask)

            #compute rank given rec_list
            with tf.variable_scope(""):
                #[batch_size, length, num_symbols]
                y_prob = tf.reshape(y_prob, [batch_size, encoder_length, num_symbols]) * \
                    tf.concat([tf.zeros([batch_size, encoder_length, 2], dtype=tf.float32), 
                                tf.ones([batch_size, encoder_length, num_symbols-2], dtype=tf.float32)], 2)
                #[batch_size, encoder_length, rec_len]
                ini_prob = tf.reshape(tf.gather_nd(y_prob, gather_idx), [batch_size, encoder_length, rec_length])
                mul_prob = ini_prob * self.rec_mask

                tmp_prob_sum = tf.expand_dims(tf.reduce_sum(mul_prob, 2), 2)
                self.prob = mul_prob / (tmp_prob_sum + 1e-18)
                #[batch_size, length, 10]
                self.values, self.index = tf.nn.top_k(self.prob, k=action_num)
                _, self.metric_index = tf.nn.top_k(self.prob, k=FLAGS.metric)

            # compute loss
            if FLAGS.use_is:
                y_beta_prob, local_beta_loss, total_beta_size, _ = sampled_sequence_loss(tf.stop_gradient(self.encoder_output, name="state_stop"), self.sessions_target, self.mask, scope="beta")

                #[batch_size, length]
                self.y_target_prob = tf.reduce_sum(y_prob * tf.one_hot(self.sessions_target, num_symbols), 2) * self.mask + (1 - self.mask)
                self.y_target_beta_prob = tf.reduce_sum(y_beta_prob * tf.one_hot(self.sessions_target, num_symbols), 2) * self.mask + (1 - self.mask)

                #[batch_size, 1]
                #self.is_coeff = tf.expand_dims(tf.reduce_prod(tf.clip_by_value((self.y_target_prob + 1e-18) / (self.y_target_beta_prob + 1e-18), 1e-10, 2), 1), 1)
                #self.is_coeff = tf.ones([batch_size, encoder_length])
                self.is_coeff = tf.clip_by_value((self.y_target_prob + 1e-18) / (self.y_target_beta_prob + 1e-18), 1e-10, 2)
                #self.is_coeff = tf.Print(self.is_coeff, ["is_coeff:",self.is_coeff, "reward:", self.reward, "mask:", self.mask, "y_target_prob:", self.y_target_prob, "y_target_beta_prob:", self.y_target_beta_prob], summarize=1e5)

                self.beta_loss = tf.reduce_sum(local_beta_loss) / total_beta_size
                self.pi_loss = tf.reduce_sum(self.is_coeff * self.reward * tf.reshape(local_loss, [batch_size, encoder_length])) / total_size
                #self.pi_loss = tf.reduce_sum(self.reward * tf.reshape(local_loss, [batch_size, encoder_length])) / total_size

                self.loss = self.beta_loss + self.pi_loss
            else:
                if True: # whether to add negative reward to non-clicked item
                    self.loss = tf.reduce_sum(tf.reshape(self.reward, [-1]) * local_loss) / total_size
                else:
                    # [batch_size, encoder_length]
                    neg_reward = 1
                    rec_reward = tf.gather_nd(tf.one_hot(self.sessions_target, num_symbols), gather_idx)
                    self.all_reward = tf.expand_dims(self.reward + neg_reward, 2) * rec_reward - neg_reward

                    _, tmp_index = tf.nn.top_k(self.prob, k=rec_length) 
                    _, tmp_tmp_index = tf.nn.top_k(tmp_index, sorted=False, k=rec_length)
                    if True: # whether to cut off the item in rec list behind the true click
                        # [batch_size, encoder_length]
                        target_pos = tf.reduce_sum(tf.cast(self.aims, dtype=tf.int32) * tmp_tmp_index, 2)
                        reward_mask = tf.reshape(tf.cumsum(tf.one_hot(target_pos, 
                                rec_length), reverse=True, axis=1), [batch_size, encoder_length, rec_length])
                        self.all_reward *= reward_mask
                    self.all_reward *= 1. / tf.cast(tmp_tmp_index + 1, dtype=tf.float32)

                    self.loss = -tf.reduce_sum((tf.reduce_sum(
                            tf.reshape(tf.gather_nd(
                                    tf.reshape(y_log_prob, [batch_size, encoder_length, num_symbols]),
                                gather_idx),
                            [batch_size, encoder_length, rec_length]) * \
                        self.all_reward * self.rec_mask, 2) / (tf.reduce_sum(self.rec_mask, 2) + 1e-12)) * self.mask) / total_size


            if FLAGS.use_active_learning:
                #[batch_size]
                self.pure_loss = tf.reduce_sum(tf.reshape(local_loss, [batch_size, encoder_length]), 1)

        #inference
        with tf.variable_scope("", reuse=True):
            self.lstm_state = tf.placeholder(tf.float32, shape=(2, 2, None, num_units))
            self.ini_state = (tf.contrib.rnn.LSTMStateTuple(self.lstm_state[0,0,:,:], self.lstm_state[0,1,:,:]), tf.contrib.rnn.LSTMStateTuple(self.lstm_state[1,0,:,:], self.lstm_state[1,1,:,:]))
            # rnn encoder   [batch_size, length, num_units]
            self.encoder_output_predict, self.encoder_state_predict = dynamic_rnn(cell, self.encoder_input, 
                    self.sessions_length, initial_state=self.ini_state, dtype=tf.float32, scope="encoder")

            tf.get_variable_scope().reuse_variables()
            #[batch_size, num_units]
            self.final_output_predict = tf.reshape(self.encoder_output_predict, [-1, num_units])
            #[batch_size, num_symbols]
            self.rec_logits = output_fn(self.final_output_predict)
            # [batch_size, action_num]
            _, self.rec_index = tf.nn.top_k(self.rec_logits[:,4:], action_num)
            self.rec_index = self.rec_index + 4

            def gumbel_max(inp, alpha, beta):
                #assert len(tf.shape(inp)) == 2
                g = tf.random_uniform(tf.shape(inp),0.0001,0.9999)
                g = -tf.log(-tf.log(g))
                inp_g = tf.nn.softmax((tf.nn.log_softmax(inp/1.0) + g * alpha) * beta)
                return inp_g            
            # [batch_size, action_num]
            _, self.random_rec_index = tf.nn.top_k(gumbel_max(self.rec_logits[:,4:], 1, 1), action_num)
            self.random_rec_index += 4

        # initialize the training process
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, 
                dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * learning_rate_decay_factor)

        self.epoch = tf.Variable(0, trainable=False, name='epoch')
        self.epoch_add_op = self.epoch.assign(self.epoch + 1)

        self.global_step = tf.Variable(0, trainable=False)
        opt = tf.train.AdamOptimizer(self.learning_rate)
        #opt = tf.train.MomentumOptimizer(self.learning_rate, 0.9) 
        #opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.params = tf.trainable_variables()
        # calculate the gradient of parameters
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


    def step_decoder(self, session, data, forward_only=False):
        input_feed = {self.sessions_input: data['sessions'],
                #self.reward: data['cum_env_dis_reward'],
                self.reward: data['cum_env_reward'],
                self.aims_idx: data['aims'],
                self.rec_lists: data['rec_lists'],
                self.rec_mask: data['rec_mask'],
                self.sessions_length: data['sessions_length']}

        '''
        outputs = session.run([self.index, self.values], input_feed)
        print data["sessions"][0]
        print data["rec_lists"][0]
        print data["purchase"][0]
        print "index:", outputs[0][0]
        print "values:", outputs[1][0]
        exit()
        '''
        if forward_only:
            output_feed = [self.loss, self.metric_index]
        else:
            output_feed = [self.loss, self.metric_index, self.gradient_norm, self.update]
        return session.run(output_feed, input_feed)


    def train(self, sess, dataset, generate_session=None, is_train=True, ftest_name="agn_test_output.txt"):
        st, ed, loss, acc, acc_1, pr_loss, pu_loss, acc_p = 0, 0, [], [], [], [], [], []
        acc_pur, acc_pur_1 = [], []
        tp, tn, fp, fn = [], [], [], []
        pur_num, pur_num_1 = 0., 0.
        #fout = open("./output.txt", "w")
        if generate_session != None:
            dataset = dataset + generate_session
        print "get %s data:len(dataset) is %d " % ("training" if is_train else "testing", len(dataset))
        if not is_train:
            fout = open(ftest_name, "w")
            fout.close()
        while ed < len(dataset):
            # print "training %.4f %%...\r" % (float(ed) / len(dataset) * 100),
            st, ed = ed, ed + FLAGS.batch_size if ed + \
                FLAGS.batch_size < len(dataset) else len(dataset)
            batch_data = gen_batched_data(dataset[st:ed])
            outputs = self.step_decoder(sess, batch_data, forward_only=False if is_train else True)
            loss.append(outputs[0])
            predict_id = outputs[1]  #[batch_size, length, 10]
            # exit()

            #print prob[0]
            tmp_acc, tmp_acc_1, pur, all_purchase, pur_1, all_purchase_1 = compute_acc(
                batch_data["aims"], predict_id, batch_data["rec_lists"], batch_data["rec_mask"], batch_data["purchase"], ftest_name=ftest_name, output=(not is_train))
            #print pur, all_purchase, pur_1, all_purchase_1
            acc.append(tmp_acc)
            acc_1.append(tmp_acc_1)
            if all_purchase != 0:
                acc_pur.append(pur / all_purchase)
            if all_purchase_1 != 0:
                acc_pur_1.append(pur_1 / all_purchase_1)
            pur_num += all_purchase
            pur_num_1 += all_purchase_1
        #print "predict:p@1:", str(np.mean(acc_1) * 100)+"%%", "p@%s:"%FLAGS.metric, str(np.mean(acc)*100)+"%%"
        print "acc in only purchase data:", np.mean(acc_pur), np.mean(acc_pur_1), pur_num, pur_num_1
        if is_train:
            sess.run(self.epoch_add_op)
        return np.mean(loss), np.mean(acc), np.mean(acc_1)



    def train_active(self, sess, dataset, generate_session=None, data_train_active_num=None, is_train=True, ftest_name="agn_active_test_output.txt"):
        st, ed, loss, acc, acc_1, pr_loss, pu_loss, acc_p = 0, 0, [], [], [], [], [], []
        acc_pur, acc_pur_1 = [], []
        tp, tn, fp, fn = [], [], [], []
        pur_num, pur_num_1 = 0., 0.
        #fout = open("./output.txt", "w")
        if generate_session != None:
            dataset = dataset + generate_session
        print "active learning, get %s data:len(dataset) is %d " % ("training" if is_train else "testing", len(dataset))
        if not is_train:
            fout = open(ftest_name, "w")
            fout.close()
        random.shuffle(dataset)
        for _ in range(len(dataset) / FLAGS.batch_size):
            sample_dataset = np.random.choice(dataset, 1000, replace=False)
            #print "epoch %d, training %.4f %%...\r" % (self.epoch.eval(session=sess), float(_) * FLAGS.batch_size / len(dataset) * 100),
            tmp_dataset = np.take(sample_dataset, self.rank_by_excute_prob(sess, sample_dataset)[:FLAGS.batch_size])
            for s in tmp_dataset:
                data_train_active_num[s[0]["num"]] += 1
            batch_data = gen_batched_data(tmp_dataset)
            outputs = self.step_decoder(sess, batch_data, forward_only=False if is_train else True)
            loss.append(outputs[0])
            predict_id = outputs[1]  #[batch_size, length, 10]

            #print prob[0]
            tmp_acc, tmp_acc_1, pur, all_purchase, pur_1, all_purchase_1 = compute_acc(
                batch_data["aims"], predict_id, batch_data["rec_lists"], batch_data["rec_mask"], batch_data["purchase"], ftest_name=ftest_name, output=(not is_train))
            #print pur, all_purchase, pur_1, all_purchase_1
            acc.append(tmp_acc)
            acc_1.append(tmp_acc_1)
            if all_purchase != 0:
                acc_pur.append(pur / all_purchase)
            if all_purchase_1 != 0:
                acc_pur_1.append(pur_1 / all_purchase_1)
            pur_num += all_purchase
            pur_num_1 += all_purchase_1
        #print "predict:p@1:", str(np.mean(acc_1) * 100)+"%%", "p@%d:"%FLAGS.metric, str(np.mean(acc)*100)+"%%"
        print "acc in only purchase data:", np.mean(acc_pur), np.mean(acc_pur_1), pur_num, pur_num_1
        if is_train:
            sess.run(self.epoch_add_op)
        return np.mean(loss), np.mean(acc), np.mean(acc_1), data_train_active_num



    def rank_by_excute_prob(self, sess, dataset):
        data_pure_loss = []
        st, ed = 0, 0
        while ed < len(dataset):
            #print "epoch %d, training %.4f %%...\r" % (epoch, float(ed) / len(dataset) * 100),
            st, ed = ed, ed + FLAGS.batch_size if ed + \
                FLAGS.batch_size < len(dataset) else len(dataset)
            data = gen_batched_data(dataset[st:ed])
            input_feed = {self.sessions_input: data['sessions'],
                    self.reward: data['cum_reward'],
                    self.rec_lists: data['rec_lists'],
                    self.rec_mask: data['rec_mask'],
                    self.sessions_length: data['sessions_length']}
            # the less the pure loss, the more the excution probability
            pure_loss = sess.run(self.pure_loss, input_feed)
            data_pure_loss.extend(pure_loss)
        sort_index = list(np.argsort(np.array(data_pure_loss)))[::-1]
        #from large to little
        return sort_index