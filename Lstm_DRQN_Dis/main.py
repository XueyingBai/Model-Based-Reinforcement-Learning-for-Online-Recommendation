# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import sys
import time
import random
import copy
random.seed(1229)
from agent import AgentModel, _START_VOCAB
from environment import EnvModel
from discriminator import DisModel

tf.app.flags.DEFINE_boolean("is_train", True, "Set to False to inference.")
tf.app.flags.DEFINE_integer("units", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("symbols", 40000, "Size of symbol list.")
tf.app.flags.DEFINE_integer("embed_units", 50, "Embedding units.")
tf.app.flags.DEFINE_integer("layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("action_num", 10, "num of recommendations")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
tf.app.flags.DEFINE_string("data_dir", "../data_session/", "Data directory")
tf.app.flags.DEFINE_string("agn_train_dir", "./agn_train", "Training directory.")
tf.app.flags.DEFINE_string("env_train_dir", "./env_train_update", "Training directory.")
tf.app.flags.DEFINE_string("dis_train_dir", "./dis_train", "Training directory.")
tf.app.flags.DEFINE_integer("per_checkpoint", 1000, "How many steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("inference_version", 36582, "The version for inferencing.")
tf.app.flags.DEFINE_boolean("log_parameters", True, "Set to True to show the parameters")
tf.app.flags.DEFINE_string("inference_path", "", "Set filename of inference, default isscreen")
tf.app.flags.DEFINE_float("min_epsilon", 1e-4, "when epsilon < this num, not random choose any longer")
tf.app.flags.DEFINE_float("max_epsilon", 0.1, "initial epsilon")
tf.app.flags.DEFINE_float("gamma", 0.9, "discount factor")
tf.app.flags.DEFINE_integer("OBSERVE", 1e6, "the num of steps before training")
tf.app.flags.DEFINE_integer("memory_size", 5000, "memory upper limit")
tf.app.flags.DEFINE_boolean("use_dis", True, "Set to True to show the parameters")
tf.app.flags.DEFINE_boolean("interact", True, "Set to True to show the parameters")

FLAGS = tf.app.flags.FLAGS

def load_data(path):
    session = {}
    output_session = []
    with open(path + "train-sci-1c.csv", "r") as fin:
        #fin.readline()
        for line in fin:
            tmp = line.strip().split(";")
            if int(tmp[0]) in session:
                session[int(tmp[0])].append({"click":tmp[2], "rec_list": tmp[3].strip().split(","), "purchase": int(tmp[4].strip())})
            else:
                session[int(tmp[0])] = [{"click":tmp[2], "rec_list": tmp[3].strip().split(","), "purchase": int(tmp[4].strip())}]
            if session[int(tmp[0])][-1]["click"] not in session[int(tmp[0])][-1]["rec_list"]:
                session[int(tmp[0])][-1]["rec_list"] += [session[int(tmp[0])][-1]["click"]]
            if session[int(tmp[0])][-1]["click"] not in session[int(tmp[0])][-1]["rec_list"]:
                session[int(tmp[0])][-1]["rec_list"] += [session[int(tmp[0])][-1]["click"]]

        skey = sorted(session.keys())

        for key in skey:
            if len(session[key]) > 1 and len(session[key]) < 40:
                output_session.append(session[key])
        print len(output_session) #(1,40)118,860    total 119,366


    return output_session

def build_vocab(data):
    print("Creating vocabulary...")
    vocab = {}
    for each_session in data:
        for item in each_session:
            for token in [item["click"]]:# + item["rec_list"]:
                if token in vocab:
                    vocab[token] += 1
                else:
                    vocab[token] = 1

    article_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    print("Loading word vectors...")

    with open("article_list.txt", "w") as fout:
        for a in article_list:
            if a in vocab:
                print >> fout, a, vocab[a]
            else:
                print >> fout, a
    if len(article_list) > FLAGS.symbols:
        article_list = article_list[:FLAGS.symbols]
    embed = []
    for i, _ in enumerate(article_list):
        if i < 4:
            embed.append(np.zeros(FLAGS.embed_units, dtype=np.float32))
        else:
            embed.append(np.random.normal(size=(FLAGS.embed_units)))
    embed = np.array(embed, dtype=np.float32)
    return article_list, embed
#**********************************************************************************

epsilon = FLAGS.max_epsilon # 0.1
def select_action(click, e_greedy):
    global epsilon
    #epsilon-greedy
    random = False
    if e_greedy and np.random.uniform() < epsilon:
        random = True
        current_action = np.random.permutation(np.arange(FLAGS.symbols)[3:])[:FLAGS.action_num]
    else:
        with agn_graph.as_default():
            #[1*len, 10]
            current_action = agn_sess.run(
                agn_model.index, feed_dict={agn_model.sessions_input: np.reshape(click, [1,1]), agn_model.lstm_state:gen_state[-1]})[-1,:]
    if e_greedy and step_index % 1e5 == 0 and epsilon > FLAGS.min_epsilon:
        print "epsilon*=0.99:", epsilon
        epsilon *= 0.99
    return np.reshape(current_action, [1, 1, FLAGS.action_num]), random

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

def rollout(state, click, rollout_list, rollout_rec_list, rollout_aim, length):
    if len(rollout_list) == length or click == 3:
        return rollout_list, rollout_rec_list, rollout_aim
    with agn_graph.as_default():
        output = agn_sess.run([agn_model.encoder_state, agn_model.rec_logits], feed_dict={agn_model.sessions_input:np.reshape(click, [1,1]), agn_model.lstm_state:state})
        next_state = output[0]
        prob = softmax(np.reshape(output[1],[FLAGS.symbols])[3:])
        try:
            action = np.reshape(np.random.choice(range(FLAGS.symbols)[3:],FLAGS.action_num,False,prob), [1,1,FLAGS.action_num])
        except ValueError as e:
            #action = np.reshape(np.take(range(FLAGS.symbols)[3:], np.argpartition(prob,-FLAGS.action_num)[-FLAGS.action_num:]), [1,1,FLAGS.action_num])
            action = np.reshape(np.random.permutation(range(FLAGS.symbols)[3:])[:FLAGS.action_num], [1,1,FLAGS.action_num])
    rollout_rec_list.append(action[0,0,:])
    with env_graph.as_default():
        #[1, 1, 10]
        output = env_sess.run([env_model.sample_index, env_model.purchase_logits], feed_dict={
            env_model.sessions_input:np.reshape(rollout_list, [1, -1]), 
            env_model.rec_lists:action, 
            env_model.rec_mask:np.ones_like(action)})

        next_click = action[0,0,output[0][0, -1, 0]]
    rollout_aim.append([0.]*FLAGS.action_num)
    rollout_aim[-1][output[0][0,-1,0]] = 1.
    rollout_list.append(next_click)

    return rollout(next_state, next_click, list(rollout_list), list(rollout_rec_list), list(rollout_aim), length)


gen_session, gen_rec_list, gen_aim, session_no, gen_loss = [], [], [], 0, [0.]
ini_state = [[[[0.]*FLAGS.units]]*2]*FLAGS.layers
gen_state = [ini_state]*2
next_session = True
rewards_recoder = []
def generate_next_click(current_click, flog, e_greedy=True):
    global gen_session, gen_rec_list, gen_aim, session_no, gen_state, next_session, rewards_recoder
    if len(gen_session) >= 5 or current_click == 3:
        gen_session = [np.random.choice(sort_start_click)]
        gen_rec_list, gen_aim = [], []
        gen_state = gen_state[1:] + [ini_state]
        session_no += 1
        next_session = True
        print >> flog, "------------next session:%d------------" % (session_no)
    else:
        gen_session.append(current_click)
        next_session = False
    click = np.reshape(np.array(gen_session), [1, len(gen_session)])
    action, random = select_action(click[0,-1], e_greedy)
    print >> flog, "current_click:", gen_session[-1],
    print >> flog, "random:", random,
    def sigmoid(x):
        return 1 / (1+ np.exp(-x))
    with agn_graph.as_default():
        current_state = gen_state[-1]
        output = agn_sess.run([agn_model.encoder_state], feed_dict={agn_model.sessions_input:np.reshape(click[-1, -1], [1,1]), agn_model.lstm_state:gen_state[-1]})
        gen_state = gen_state[1:] + [output[0]]
        next_state = gen_state[-1]

    with env_graph.as_default():
        #[1, 1, 10]
        output = env_sess.run([env_model.sample_index, env_model.purchase_logits], feed_dict={env_model.sessions_input:click, env_model.rec_lists:action, env_model.rec_mask:np.ones_like(action)})
        next_click = action[0, 0, output[0][0, -1, 0]]
        prob = sigmoid(output[1][0,-1])
        reward = 4 if output[1][0, -1] > 0 else 1
        #reward = 4 * prob + (1 - prob)
        print >> flog, "next_click:", next_click, "reward:", reward, "prob:", prob,
        gen_rec_list.append(list(action[0,0,:]))
        gen_aim.append([0.]*FLAGS.action_num)
        gen_aim[-1][output[0][0,-1,0]] = 1.
    if FLAGS.use_dis:
        with dis_graph.as_default():
            score = []
            length = 5 if len(gen_session) != 5 else 1
            for _ in range(length):
                tmp_total_click, tmp_total_rec_list, tmp_total_aim = rollout(current_state,current_click,list(gen_session), list(gen_rec_list), list(gen_aim), 5)
                prob = dis_sess.run(dis_model.prob, {
                    dis_model.sessions_input:np.reshape(tmp_total_click, [1, -1]),
                    dis_model.sessions_length:np.array([len(tmp_total_click)]),
                    dis_model.rec_lists:np.array([tmp_total_rec_list]),
                    dis_model.rec_mask:np.ones([1,len(tmp_total_click),FLAGS.action_num]),
                    dis_model.aims:np.reshape(tmp_total_aim, [1, len(tmp_total_click), FLAGS.action_num])
                    })
                score.append(prob[0, 1])

            norm_score = np.mean(score) * 2 - 1
            reward += norm_score
        print >> flog, "score:%.4f"%norm_score,

        with env_graph.as_default():
            output = env_sess.run([env_model.score_loss, env_model.norm_prob, env_model.gradient_norm, env_model.update], 
                    feed_dict={env_model.sessions_input:click, 
                            env_model.rec_lists:action, 
                            env_model.rec_mask:np.ones_like(action), 
                            env_model.score:np.reshape(np.mean(score), [1])
                            })
            gen_loss.append(output[0])
            rewards_recoder.append(np.mean(score))
            #gen_state = gen_state[1:] + [agn_sess.run(agn_model.encoder_state, feed_dict={agn_model.sessions_input:np.reshape(next_click, [1, 1]), agn_model.lstm_state:gen_state[-1]})]
    print >> flog, "action:", list(action[0,0,:])
    return gen_session[-1], next_click, current_state, next_state, action[0, 0, :], reward

pool_size = 131072
step_index, generate_session = 0, []
def generate_data(size, flog):
    global step_index, generate_session, current_click, session_no, next_session
    tmp_generate_session = []
    tmp_session_no = session_no
    current_click = np.random.choice(sort_start_click)
    while session_no < tmp_session_no + size:
        step_index += 1
        #print "generating %.4f %%\r" % (float(_)  / size * 100),
        current_click, next_click, current_state, next_state, current_action, reward = generate_next_click(current_click, flog)
        if not next_session and len(tmp_generate_session) > 0:
            tmp_generate_session[-1].append({"session_no":session_no, "current_click":current_click, "next_click": next_click, "current_state": current_state, "next_state": next_state, "action": current_action, "aim_action":[next_click], "reward": reward})
        else:
            tmp_generate_session.append([{"session_no":session_no, "current_click":current_click, "next_click": next_click, "current_state": current_state, "next_state": next_state, "action": current_action, "aim_action":[next_click], "reward": reward}])
        #generate_session.append({"session_no": session_no, "current_click":current_click, "next_click": next_click, "current_state": current_state, "next_state": next_state, "action": current_action, "aim_action":[next_click], "reward": reward})
        current_click = next_click
    for _, s in enumerate(tmp_generate_session[:-1]):
        for i in range(len(s)-1):
            s[i]["reward"] = s[i+1]["reward"]
            generate_session.append(s[i])
        s[-1]["reward"] = 0# if s[-1]["next_click"] == 3 else 1
        generate_session.append(s[-1])
    if len(generate_session) > pool_size:
        generate_session = generate_session[-pool_size:]
#**********************************************************************************

def gen_batched_data(data, max_len, max_rec_len):
    sessions, rec_lists, aims, sessions_length, purchase, rec_mask = [], [], [], [], [], []
    aim_rec_lists, aim_rec_mask = [], []
    def padding(sent, l):
        return sent + [3] + [0] * (l-len(sent)-1)
    def padding_list(vec, l):
        return vec + [-1.] * (l - len(vec))
    def get_vec(session_al):
        session_tmp = []
        mask_tmp = []
        for al in session_al:
            session_tmp.append(al + [0] * (max_rec_len - len(al)))
            mask_tmp.append([1.] * len(al) + [0.] * (max_rec_len - len(al)))
        session_tmp += [[0]*max_rec_len] * (max_len-len(session_tmp))
        mask_tmp += [[0.]*max_rec_len] * (max_len-len(mask_tmp))
        return session_tmp, mask_tmp

    def get_aim(session_aim, rec_list):
        s_aim = []
        for a, rlist in zip(session_aim, rec_list):
            tmp = [0.] * max_rec_len
            for i, r in enumerate(rlist):  
                if a == r:
                    tmp[i] = 1.
                    break
            s_aim.append(tmp)
        return s_aim

    for item in data:
        sessions.append(padding([s["click"] for s in item], max_len))    
        purchase.append(padding_list([s["purchase"] * 3 + 1 for s in item] + [0.], max_len))
        rl, rm = get_vec([s["rec_list"] for s in item][1:] + [[3]])
        al, am = get_vec([[s["click"]] for s in item][1:] + [[3]])
        rec_lists.append(rl)
        rec_mask.append(rm)
        aim_rec_lists.append(al)
        aim_rec_mask.append(am)
        sessions_length.append(len(item)+1)
        aims.append(get_aim(sessions[-1][1:] + [0], rl))

    batched_data = {'sessions': np.array(sessions),
            'rec_lists': np.array(rec_lists),
            'rec_mask' : np.array(rec_mask),
            'aim_rec_lists': np.array(aim_rec_lists),
            'aim_rec_mask' : np.array(aim_rec_mask),
            'purchase': np.array(purchase),
            'aims': np.array(aims),
            'sessions_length': sessions_length}
    #np.set_printoptions(threshold=np.inf)
    #for key in batched_data:
        #print key, np.shape(batched_data[key]), batched_data[key][:3]
    #exit()
    return batched_data

def train(dataset, is_train=True):
    st, ed = 0, 0
    click_state, next_click_state, click, next_click, reward, action, action_mask, aim_action, aim_action_mask = [], [], [], [], [], [], [], [], []
    max_rec_len = max([len(s["rec_list"]) for item in dataset for s in item]) 

    while ed < len(dataset):
        #print "epoch %d, training generate state %.4f %%...\r" % (epoch, float(ed) / len(dataset) * 100),
        st, ed = ed, ed + FLAGS.batch_size if ed + \
            FLAGS.batch_size < len(dataset) else len(dataset)
        max_len = max([len(item) for item in dataset[st:ed]]) + 1 
        batch_data = gen_batched_data(dataset[st:ed], max_len, max_rec_len)

        states = [[[[[0.]*FLAGS.units]*(ed-st)]*2]*FLAGS.layers]
        for i in range(np.shape(batch_data["sessions"])[1]):
            states.append(agn_sess.run(agn_model.encoder_state, feed_dict={agn_model.sessions_input:np.reshape(batch_data["sessions"][:,i], [-1, 1]), agn_model.lstm_state:states[-1]}))
            #print np.shape(states[-1]), np.shape(states[-1][0][0]), np.shape(states[-1][0][1]), np.shape(states[-1][1][0]), np.shape(states[-1][1][1])
            #print "---------------------------------------------"
        #[length, num_layers, 2, batch_size, num_units] -> [batch_size, length, num_layers, 2, 1, num_units]        
        states = np.expand_dims(np.transpose(states, [3, 0, 1, 2, 4]), 4)
        # [batch_size, length, num_units]
        batch_states = states

        for bs, bc, brl, brm, bp, bsl, barl, barm in zip(batch_states, batch_data["sessions"], batch_data["rec_lists"], batch_data["rec_mask"], batch_data["purchase"], batch_data["sessions_length"], batch_data["aim_rec_lists"], batch_data["aim_rec_mask"]):
            # bs:[length, units], brl:[length, 10], brm:[length, 10], ba:[length, ]
            purchase = False
            if 1 in bp:
                purchase = True
            for i in range(1, bsl):
                click_state.append(bs[i-1])
                next_click_state.append(bs[i])
                click.append(bc[i-1])
                next_click.append(bc[i])
                action.append(brl[i-1])
                action_mask.append(brm[i-1])
                aim_action.append(barl[i-1])
                aim_action_mask.append(barm[i-1])
                #reward.append(bp[i-1] * 3 + 1)
                reward.append(bp[i])
    new_click_state, new_next_click_state, new_click, new_next_click, new_reward, new_action, new_action_mask, new_aim_action, new_aim_action_mask = \
                click_state, next_click_state, click, next_click, reward, action, action_mask, aim_action, aim_action_mask
    st, ed, loss, acc_list, acc_1_list = 0, 0, [], [], []
    pur_acc_list, pur_acc_1_list = [], []
    pur_num, pur_num_1 = 0., 0.
    while ed < len(new_click_state):
        #print "epoch %d, training %.4f %%...\r" % (epoch, float(ed) / len(new_click_state) * 100),
        st, ed = ed, ed + FLAGS.batch_size if ed + \
            FLAGS.batch_size < len(new_click_state) else len(new_click_state)
        click_state, next_click_state, click, next_click, reward, action, action_mask, aim_action, aim_action_mask = \
            new_click_state[st:ed], new_next_click_state[st:ed], new_click[st:ed], new_next_click[st:ed], new_reward[st:ed], new_action[st:ed], new_action_mask[st:ed], new_aim_action[st:ed], new_aim_action_mask[st:ed]

        if FLAGS.interact and is_train:
            '''
            data_num = (ed - st) / 3 * 2
            generate_data(data_num, flog)
            replay_buffer = list(np.random.permutation(generate_session[:-data_num]))[:data_num/2] + generate_session[-data_num:]
            ''' 
            length = len(generate_session)
            if st % length < ed % length:
                replay_buffer = generate_session[st % length : ed % length]
            else:
                replay_buffer = generate_session[st % length : ] + generate_session[:ed % length]
            #[batch_size, num_layers, 2, 1, num_units] -> [batch_size, num_layers, 2, num_units] -> [num_layers, 2, batch_size, num_units]
            next_click_state = np.concatenate([next_click_state, [r["next_state"] for r in replay_buffer]], 0)
            click_state = np.concatenate([click_state, [r["current_state"] for r in replay_buffer]], 0)

            next_click = np.concatenate([next_click, [r["next_click"] for r in replay_buffer]], 0)
            click = np.concatenate([click, [r["current_click"] for r in replay_buffer]], 0)

            reward = np.concatenate([reward, [r["reward"] for r in replay_buffer]], 0)
            for r in replay_buffer:
                aim_action = np.concatenate([aim_action, np.reshape(list(r["aim_action"]) + [0] * (max_rec_len-len(r["aim_action"])), [1, max_rec_len])], 0)    
                aim_action_mask = np.concatenate([aim_action_mask, np.reshape(list(np.ones_like(r["aim_action"])) + [0.] * (max_rec_len-len(r["aim_action"])), [1, max_rec_len])], 0)
                action = np.concatenate([action, np.reshape(list(r["action"]) + [0] * (max_rec_len-len(r["action"])), [1, max_rec_len])], 0)    
                action_mask = np.concatenate([action_mask, np.reshape(list(np.ones_like(r["action"])) + [0.] * (max_rec_len-len(r["action"])), [1, max_rec_len])], 0)

        click_state = np.transpose(np.reshape(click_state, [np.shape(click_state)[0],np.shape(click_state)[1],np.shape(click_state)[2],np.shape(click_state)[4]]), [1,2,0,3])
        next_click_state = np.transpose(np.reshape(next_click_state, [np.shape(next_click_state)[0],np.shape(next_click_state)[1],np.shape(next_click_state)[2],np.shape(next_click_state)[4]]), [1,2,0,3])

        q_next = agn_sess.run(agn_model.rec_logits, feed_dict={agn_model.lstm_state: next_click_state, agn_model.sessions_input: np.reshape(next_click, [-1,1])})
        q = agn_sess.run(agn_model.rec_logits, feed_dict={agn_model.lstm_state: click_state, agn_model.sessions_input: np.reshape(click, [-1,1])})

        acc, acc_1, all_num_10, all_num_1, pur, pur_1, all_purchase_10, all_purchase_1 = 0., 0., 0., 0., 0., 0., 0., 0.
        range_len = len(action) - len(replay_buffer) if is_train else len(action)
        for i in range(range_len):
            #rec_idx = np.argpartition(q[i],-FLAGS.action_num)[-FLAGS.action_num:]
            rec_list = []
            for a, m in zip(action[i], action_mask[i]):
                if m == 0:
                    break
                rec_list.append(a)
            rec_score = np.take(q[i], rec_list)
            sort_idx = np.argsort(rec_score)

            if rec_list[0] != 3:
                rec_idx_10 = rec_list
                if len(rec_score) > 10:
                    all_num_10 += 1
                    rec_idx_10 = np.take(rec_list, sort_idx[-10:])  
                    '''
                    random_rec_idx_10 = np.random.permutation(rec_list)[:10]
                    if next_click[i] in random_rec_idx_10:
                        random_acc += 1.
                    '''
                    if next_click[i] in rec_idx_10:
                        acc += 1.
                    
                    if reward[i] == 4:
                        all_purchase_10 += 1
                        if next_click[i] in rec_idx_10:
                            pur += 1

                if len(rec_score) > 1:
                    all_num_1 += 1
                    rec_idx = rec_list[sort_idx[-1]]
                    if rec_idx == 1 or rec_idx == 3:
                        rec_idx = rec_list[sort_idx[-2]]
                        if (rec_idx == 1 or rec_idx == 3) and len(sort_idx) > 2:
                            rec_idx = rec_list[sort_idx[-3]]

                    '''
                    random_rec_idx = np.random.permutation(rec_list)[0]                    
                    if next_click[i] == random_rec_idx:
                        random_acc_1 += 1.
                    '''
                    if next_click[i] == rec_idx:
                        acc_1 += 1.

                    if reward[i] == 4:
                        all_purchase_1 += 1
                        if next_click[i] == rec_idx:
                            pur_1 += 1

        acc_list.append(acc / all_num_10)
        acc_1_list.append(acc_1 / all_num_1)        
        if all_purchase_10 != 0:
            pur_acc_list.append(pur / all_purchase_10)
        if all_purchase_1 != 0:
            pur_acc_1_list.append(pur_1 / all_purchase_1)
        pur_num += all_purchase_10
        pur_num_1 += all_purchase_1
        #print pur, all_purchase_10, pur_1, all_purchase_1
        q_target = []
        for i in range(len(reward)):
            #q_next:[batch, action_num]
            #q_target.append(reward[i] + FLAGS.gamma * np.sum(q_next[i]))
            q_target.append(reward[i] + FLAGS.gamma * np.max(q_next[i]))
            #q_target.append(reward[i] + FLAGS.gamma * np.sum([q_next[i, k] for k in np.argpartition(q_next[i],-FLAGS.action_num)[-FLAGS.action_num:]]))

        if is_train:
            output_feed = [agn_model.loss, agn_model.gradient_norm, agn_model.update]
        else:
            output_feed = [agn_model.loss]
        l = agn_sess.run(output_feed,
                    feed_dict={agn_model.lstm_state: click_state,
                                agn_model.sessions_input: np.reshape(click, [-1,1]),
                                agn_model.action: aim_action,
                                agn_model.action_mask: aim_action_mask,
                                agn_model.q_target: np.array(q_target)})
        loss.append(l[0])
    if is_train:
        agn_sess.run(agn_model.epoch_add_op)
    print "\nacc in only purchase data:", np.mean(pur_acc_list), np.mean(pur_acc_1_list), pur_num, pur_num_1
    return np.mean(loss), np.mean(acc_list), np.mean(acc_1_list)

def dis_gen_batched_data(data, label, max_len, max_rec_len):
    batched_data = gen_batched_data(data, max_len, max_rec_len)
    batched_data["labels"] = np.array([label]*len(data))
    return batched_data

def train_dis(dataset, is_train=True):
    st, ed, loss, acc = 0, 0, [], []
    replay_buffer = {}

    if is_train:
        generate_session_train = generate_session[:len(generate_session)*19/20]
    else:
        generate_session_train = generate_session[-len(generate_session)/20:]

    for item in generate_session_train:
        if item["session_no"] in replay_buffer:
            #session[int(tmp[0])].append({"click":aid2index[tmp[2]] if tmp[2] in aid2index else 1, "rec_list": [aid2index[rl] if rl in aid2index else 1 for rl in tmp[3].strip().split(",")], "purchase": int(tmp[4].strip())})
            replay_buffer[item["session_no"]].append({"click":item["current_click"], "rec_list":list(item["action"]), "purchase":0.})
        else:
            #session[int(tmp[0])] = [{"click":aid2index[tmp[2]] if tmp[2] in aid2index else 1, "rec_list": [aid2index[rl] if rl in aid2index else 1 for rl in tmp[3].strip().split(",")], "purchase": int(tmp[4].strip())}]
            replay_buffer[item["session_no"]] = [{"click":item["current_click"], "rec_list":list(item["action"]), "purchase":0.}]
    min_no = min([key for key in replay_buffer])
    max_no = max([key for key in replay_buffer])
    replay_buffer.pop(min_no)
    if max_no in replay_buffer and len(replay_buffer) > 1:
        replay_buffer.pop(max_no)

    data_gen = []
    for r in replay_buffer:
        data_gen.append(replay_buffer[r])
    length = len(data_gen)
    max_rec_len = max([len(s["rec_list"]) for item in dataset for s in item] + [len(s["rec_list"]) for item in data_gen]) 
    while ed < len(dataset):
        #print "dis_epoch %d, training %.4f %%...\r" % (dis_epoch, float(ed) / len(dataset) * 100),
        st, ed = ed, ed + FLAGS.batch_size if ed + \
            FLAGS.batch_size < len(dataset) else len(dataset)

        tmp_data_gen = []
        if st % length < ed % length:
            tmp_data_gen = data_gen[st % length : ed % length]
        else:
            tmp_data_gen = data_gen[st % length : ] + data_gen[:ed % length]
        max_len = max([len(item) for item in dataset[st:ed]] + [len(item) for item in tmp_data_gen]) + 1
        batch_data_1 = dis_gen_batched_data(dataset[st:ed], 1, max_len, max_rec_len)
        batch_data_2 = dis_gen_batched_data(tmp_data_gen, 0, max_len, max_rec_len)
        batch_data = {}
        for key in batch_data_1:
            batch_data[key] = np.concatenate([batch_data_1[key], batch_data_2[key]],0)
        if is_train:
            outputs = dis_model.step_decoder(dis_sess, batch_data)
        else:
            outputs = dis_model.step_decoder(dis_sess, batch_data, forward_only=True)
        loss.append(outputs[0])
        acc.append(outputs[1])

    if is_train:
        dis_sess.run(dis_model.epoch_add_op)

    return np.mean(loss), np.mean(acc)

#**********************************************************************************
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
env_graph = tf.Graph()
agn_graph = tf.Graph()
dis_graph = tf.Graph()
env_sess = tf.Session(config=config, graph=env_graph)
agn_sess = tf.Session(config=config, graph=agn_graph)
dis_sess = tf.Session(config=config, graph=dis_graph)


data = load_data(FLAGS.data_dir)
#random_select = np.random.permutation(np.arange(len(data)))
#random_select = range(len(data))
vocab, embed = build_vocab(data)
aid2index = {}
index2aid = {}
for i,a in enumerate(vocab):
    aid2index[a] = i
    index2aid[i] = a

data_train = []
for s in data:
    tmps = []
    for c in s:
        c["click"] = aid2index[c["click"]] if c["click"] in aid2index else 1
        c["rec_list"] = list(set([aid2index[rl] if rl in aid2index else 1 for rl in c["rec_list"]])) + [3]
        tmps.append(c)
    data_train.append(tmps)

data = data_train

start_click = {}
for d in data:
    k = d[0]["click"]
    if k in start_click:
        start_click[k] += 1
    else:
        start_click[k] = 1
sort_start_click = sorted(start_click, key=start_click.get, reverse=True)[1:10000]

data_train = data[:(len(data) / 20) * 19]
data_dev = data[(len(data) / 20) * 19:]

'''
with open("train_data.txt", "w") as fout:
    for k, s in enumerate(data_train):
        print >> fout, "------------next session:%d------------"%k
        for i in range(1, len(s)):
            print >> fout, "current_click:", s[i-1]["click"], "next_click:", s[i]["click"], "purchase:", s[i-1]["purchase"], "action:", s[i]["rec_list"]
exit()
'''
print np.mean([len(s) for s in data_train]), np.mean([len(s) for s in data_dev])
if FLAGS.interact:
    with env_graph.as_default():
        env_model = EnvModel(
                FLAGS.symbols,
                FLAGS.embed_units,
                FLAGS.units,
                FLAGS.layers,
                is_train=True,
                vocab=vocab,
                embed=embed)
        env_model.print_parameters()
        env_model.saver.restore(env_sess, tf.train.latest_checkpoint(FLAGS.env_train_dir))
        print("Reading environment model parameters from %s" % FLAGS.env_train_dir)

with agn_graph.as_default():
    agn_model = AgentModel(
            FLAGS.symbols,
            FLAGS.embed_units,
            FLAGS.units,
            FLAGS.layers,
            is_train=True,
            embed=embed,
            action_num=FLAGS.action_num)
    agn_model.print_parameters()
    #agn_model.saver.restore(agn_sess, tf.train.latest_checkpoint(FLAGS.agn_train_dir))
    if tf.train.get_checkpoint_state(FLAGS.agn_train_dir):
        print("Reading agent model parameters from %s" % FLAGS.agn_train_dir)
        agn_model.saver.restore(agn_sess, tf.train.latest_checkpoint(FLAGS.agn_train_dir))
    else:
        print("Created agent model with fresh parameters.")
        agn_sess.run(tf.global_variables_initializer())
if FLAGS.use_dis:
    with dis_graph.as_default():
        dis_model = DisModel(
                FLAGS.symbols,
                FLAGS.embed_units,
                FLAGS.units,
                FLAGS.layers,
                is_train=True,
                vocab=vocab,
                embed=embed)
        dis_model.print_parameters()
        #agn_model.saver.restore(agn_sess, tf.train.latest_checkpoint(FLAGS.dis_train_dir))
        if tf.train.get_checkpoint_state(FLAGS.dis_train_dir):
            print("Reading dis model parameters from %s" % FLAGS.dis_train_dir)
            dis_model.saver.restore(dis_sess, tf.train.latest_checkpoint(FLAGS.dis_train_dir))
        else:
            print("Created dis model with fresh parameters.")
            dis_sess.run(tf.global_variables_initializer())


#pre_losses = [1e18] * 3
while True:    
    if FLAGS.interact:
        start_time = time.time()
        flog = open("%s/train_log_%d.txt"%(FLAGS.agn_train_dir, session_no), "w")
        generate_data(1000, flog)
        flog.close()        
        with open("./generate_session.txt", "w") as fout:
            key = ["session_no", "current_click", "next_click", "action", "reward"]
            print >> fout, " ".join(key)
            for s in generate_session:
                for k in key:
                    print >> fout, s[k],
                print >> fout
        print "generate-time %.4f" % (time.time()-start_time)
    with agn_graph.as_default():
        epoch = agn_model.epoch.eval(session=agn_sess)
        #random.shuffle(data_train)
        start_time = time.time()
        loss, acc, acc_1 = train(data_train)
        agn_model.saver.save(agn_sess, '%s/checkpoint' % FLAGS.agn_train_dir, global_step=agn_model.global_step)
        #if loss > max(pre_losses):  # Learning rate decay
            #agn_sess.run(agn_model.learning_rate_decay_op)
        #pre_losses = pre_losses[1:] + [loss]
        print "epoch %d learning rate %.4f epoch-time %.4f perplexity [%.8f] p@10 %.4f%% p@1 %.4f%%" \
                % (epoch, agn_model.learning_rate.eval(session=agn_sess), time.time() - start_time, loss, acc*100, acc_1*100)
        loss, acc, acc_1 = train(data_dev, is_train=False)
        print "        test_set, perplexity [%.8f] p@10 %.4f%% p@1 %.4f%%" % (loss, acc*100, acc_1*100)
        print "--------------------------------------"

    if FLAGS.use_dis:
        with env_graph.as_default():
            env_epoch = env_model.epoch.eval(session=env_sess)
            env_model.saver.save(env_sess, '%s/checkpoint' % FLAGS.env_train_dir, global_step=env_model.global_step)
            env_sess.run(env_model.epoch_add_op)
            print "env_epoch %d learning rate %.4f epoch-time %.4f env_loss [%.8f] env_rewards [%.8f]" \
                    % (env_epoch, env_model.learning_rate.eval(session=env_sess), time.time() - start_time, np.mean(gen_loss), np.mean(rewards_recoder))
            gen_loss, rewards_recoder = [], []
            print "--------"

        with dis_graph.as_default():
            dis_epoch = dis_model.epoch.eval(session=dis_sess)
            #random.shuffle(data_train)
            start_time = time.time()
            loss, acc = train_dis(data_train)
            dis_model.saver.save(dis_sess, '%s/checkpoint' % FLAGS.dis_train_dir, global_step=dis_model.global_step)
            #if loss > max(pre_losses):  # Learning rate decay
                #agn_sess.run(agn_model.learning_rate_decay_op)
            #pre_losses = pre_losses[1:] + [loss]
            print "dis_epoch %d learning rate %.4f epoch-time %.4f perplexity [%.8f] acc %.4f%%" \
                    % (dis_epoch, dis_model.learning_rate.eval(session=dis_sess), time.time() - start_time, loss, acc)
            loss, acc = train_dis(data_dev, is_train=False)
            print "        test_set, perplexity [%.8f] acc %.4f%% " % (loss, acc*100)

        print "*********************************************************************"

    #with env_graph.as_default():
        #env_model.saver.restore(env_sess, tf.train.latest_checkpoint(FLAGS.env_train_dir))
