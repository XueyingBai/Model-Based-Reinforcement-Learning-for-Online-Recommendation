# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import sys
import time
import random
import copy
random.seed(1229)
from agent import AgentModel
from actor_critic import acAgentModel
from environment import EnvModel
from discriminator import DisModel
from utils import FLAGS, sigmoid, softmax, load_data, build_vocab, gen_batched_data
import os
from tensorflow.python import pywrap_tensorflow
#**********************************************************************************

fout = open("./agn_test_output.txt", "w")
fout.close()
fout = open("./env_test_output.txt", "w")
fout.close()
FLAGS.interact_data_dir = FLAGS.interact_data_dir.replace("random_prob", "random_prob%.1f"%FLAGS.random_prob)
FLAGS.agn_train_dir = FLAGS.agn_train_dir.replace("random_prob", "random_prob%.1f_%d"%(FLAGS.random_prob, FLAGS.data_size))
FLAGS.env_train_dir = FLAGS.env_train_dir.replace("random_prob", "random_prob%.1f_%d"%(FLAGS.random_prob, FLAGS.data_size))
FLAGS.dis_train_dir = FLAGS.dis_train_dir.replace("random_prob", "random_prob%.1f_%d"%(FLAGS.random_prob, FLAGS.data_size))
FLAGS.data_name = FLAGS.data_name.replace("random", "random%.1f"%FLAGS.random_prob).replace(".txt", "_%d.txt"%FLAGS.data_size)
if not os.path.exists(FLAGS.interact_data_dir):
    os.makedirs(FLAGS.interact_data_dir)
if not os.path.exists(FLAGS.agn_train_dir):
    os.makedirs(FLAGS.agn_train_dir)
if not os.path.exists(FLAGS.env_train_dir):
    os.makedirs(FLAGS.env_train_dir)
if not os.path.exists(FLAGS.dis_train_dir):
    os.makedirs(FLAGS.dis_train_dir)

gen_session, gen_rec_list, gen_aims_idx, gen_purchase, session_no = [], [], [], [], 0
ini_state = [[[[0.]*FLAGS.units]]*2]*FLAGS.layers
gen_state = ini_state

next_session = True

def select_action(click, state):
    current_action = [aid2index[item] for item in list(np.random.permutation(vocab[4:])[:FLAGS.action_num])]
    # encoder_state = state
    # return np.concatenate([np.reshape(current_action, [1, 1, FLAGS.action_num])[:,:,:-1], np.reshape([3], [1,1,1])], 2), encoder_state
    with agn_graph.as_default():
        #[1*len, 10]
        output = agn_sess.run(
            [agn_model.rec_index, agn_model.encoder_state_predict], 
            feed_dict={agn_model.sessions_input: np.reshape(click, [1,1]), 
                    agn_model.sessions_length: np.array([1]),
                    agn_model.lstm_state:state})
    current_action = output[0]
    encoder_state = output[1]
    # return np.concatenate([np.reshape(current_action, [1, 1, FLAGS.action_num]), np.reshape([3], [1,1,1])], 2), encoder_state
    return np.reshape(current_action, [1, 1, FLAGS.action_num]), encoder_state
    # if random.random() < 1-1e-18:
    #     return np.concatenate([np.reshape(current_action, [1, 1, FLAGS.action_num]), np.reshape([3], [1,1,1])], 2), encoder_state
    # else:
    #     return np.reshape(current_action, [1, 1, FLAGS.action_num]), encoder_state

def rollout(state, click, rollout_list, rollout_rec_list, rollout_aim, rollout_purchase, length):
    rollout_list.append(click)

    with agn_graph.as_default():
        output = agn_sess.run([agn_model.encoder_state_predict, agn_model.random_rec_index], feed_dict={
            agn_model.sessions_input:np.reshape(click, [1,1]),
            agn_model.sessions_length:[1],
            agn_model.lstm_state:state})
        next_state = output[0]
        # action = np.concatenate([np.reshape(output[1], [1, 1, FLAGS.action_num]), np.reshape([3], [1,1,1])], 2)
        action = np.reshape(output[1], [1, 1, FLAGS.action_num])
        # if random.random() < 0.5:
        #     action = np.concatenate([np.reshape(output[1], [1, 1, FLAGS.action_num]), np.reshape([3], [1,1,1])], 2)
        # else:
        #     action = np.reshape(output[1], [1, 1, FLAGS.action_num])
        rollout_rec_list.append(action[0,0,:])

    with env_graph.as_default():
        #[1, 1, 10]
        rec_list = np.reshape(rollout_rec_list[-1], [1,1,-1])
        output = env_sess.run([env_model.inf_argmax_index, env_model.inf_purchase_logits], feed_dict={
            env_model.sessions_input:np.reshape(rollout_list, [1, -1]), 
            env_model.rec_lists:rec_list, 
            env_model.rec_mask:np.ones_like(rec_list),
            env_model.sessions_length:[len(rollout_list)]})

        next_click = rec_list[0,0,output[0][0, -1, 0]]
        if FLAGS.use_simulated_data:
            rollout_purchase.append(output[1][0,-1])
        else:
            rollout_purchase.append(sigmoid(output[1][0,-1]))
        rollout_aim.append(output[0][0,-1,0])

    if len(rollout_list) >= length or click == 3:
        return rollout_list, rollout_rec_list, rollout_aim, rollout_purchase
    return rollout(next_state, next_click, list(rollout_list), list(rollout_rec_list), list(rollout_aim), list(rollout_purchase), length)


def generate_next_click(current_click, flog, use_dis=FLAGS.use_dis):
    global gen_session, gen_rec_list, gen_aims_idx, gen_state, gen_purchase
    global session_no, next_session

    if len(gen_session) >= max_interact_len or current_click == 3:
        gen_session = [np.random.choice(sort_start_click)]
        gen_rec_list, gen_aims_idx, gen_purchase = [], [], []
        gen_state = ini_state
        session_no += 1
        next_session = True
        current_click = gen_session[-1]
        print >> flog, "------------next session:%d------------" % (session_no)
    else:
        gen_session.append(current_click)
        next_session = False
    session_click = np.reshape(np.array(gen_session), [1, len(gen_session)])
    action, state = select_action(session_click[0,-1], gen_state)
    print >> flog, "current_click:", current_click,
    gen_state = state

    with env_graph.as_default():
        #[1, 1, 10]
        output = env_sess.run([env_model.inf_argmax_index, env_model.inf_purchase_logits], feed_dict={
            env_model.sessions_input:session_click, 
            env_model.rec_lists:action, 
            env_model.rec_mask:np.ones_like(action),
            env_model.sessions_length:[len(session_click[0])]})
        next_click = action[0, 0, output[0][0, -1, 0]]
        if FLAGS.use_simulated_data:
            purchase_prob = output[1][0,-1]
        else:
            purchase_prob = sigmoid(output[1][0,-1])
        print >> flog, "next_click:", next_click, "purchase_prob:", purchase_prob, "reward:", 4 if output[1][0, -1] > 0 else 1,
        gen_rec_list.append(list(action[0,0,:]))
        gen_aims_idx.append(output[0][0,-1,0])
        gen_purchase.append(purchase_prob)
    dis_reward = 1.

    if use_dis:
        with dis_graph.as_default():
            score = []
            rollout_num = 5 if (len(gen_session) < max_interact_len) and (next_click != 3) else 1
            for _ in range(rollout_num):
                tmp_total_click, tmp_total_rec_list, tmp_total_aims_idx, tmp_total_purchase = rollout(gen_state,next_click,list(gen_session), list(gen_rec_list), list(gen_aims_idx), list(gen_purchase), max_interact_len+1)
                # print current_click,list(gen_session), list(gen_rec_list), list(gen_purchase)
                # print np.reshape(tmp_total_click, [1, -1])
                # print np.array([len(tmp_total_click)])
                # print np.array([tmp_total_rec_list])
                # print np.reshape(tmp_total_aims_idx, [1, len(tmp_total_click)])
                # print np.reshape(tmp_total_purchase, [1, len(tmp_total_purchase)])
                # print "--------------"
                prob = dis_sess.run(dis_model.prob, {
                    dis_model.sessions_input:np.reshape(tmp_total_click, [1, -1]),
                    dis_model.sessions_length:np.array([len(tmp_total_click)]),
                    dis_model.rec_lists:np.array([tmp_total_rec_list]),
                    dis_model.rec_mask:np.ones([1,len(tmp_total_click),len(tmp_total_rec_list[-1])]),
                    dis_model.aims_idx:np.reshape(tmp_total_aims_idx, [1, len(tmp_total_click)]),
                    dis_model.purchase:np.reshape(tmp_total_purchase, [1, len(tmp_total_purchase)])
                    })
                score.append(prob[0])
            norm_score = np.mean(score)
            # print "score:", score, norm_score

        dis_reward = norm_score
        print >> flog, "dis_reward:%.8f" % dis_reward,

        #print session_click, action, np.ones_like(action), np.reshape([norm_score], [1, 1]), np.reshape(gen_aims[-1], [1,-1]),[len(session_click[0])]
        #print "--------------------------------------------------------"

    action = list(action[0,0,:])
    # if 3 not in action:
    #     action.append(3)
    print >> flog, "action:", action
    return current_click, next_click, action, purchase_prob, dis_reward

generate_session =[]
def generate_data(size, flog, use_dis=FLAGS.use_dis):
    global generate_session, current_click, session_no, next_session
    tmp_session_no = session_no
    current_click = np.random.choice(sort_start_click)
    while session_no < tmp_session_no + size:
        # print "generating %.4f %%...\r" % (float(session_no-tmp_session_no) / size * 100),
        current_click, next_click, current_action, purchase_prob, dis_reward = generate_next_click(current_click, flog, use_dis=use_dis)
        if not next_session and len(generate_session) > 0:
            generate_session[-1].append({"session_no":session_no, "click":current_click, "rec_list": current_action, "purchase":(0. if purchase_prob<=0.5 else 1.), "dis_reward": dis_reward})
        else:
            if len(generate_session) > 0:
                length = len(generate_session[-1])
                for i in range(1, length):
                    generate_session[-1][length-i]["rec_list"] = generate_session[-1][length-i-1]["rec_list"]
                    generate_session[-1][length-i]["purchase"] = generate_session[-1][length-i-1]["purchase"]
                generate_session[-1][0]["rec_list"] = [generate_session[-1][0]["click"]]
                generate_session[-1][0]["purchase"] = 0.

            generate_session.append([{"session_no":session_no, "click":current_click, "rec_list": current_action, "purchase":(0. if purchase_prob<=0.5 else 1.), "dis_reward": dis_reward}])
        current_click = next_click
    next_session = True
    if len(generate_session) > max_pool_size:
        generate_session = generate_session[-max_pool_size:]
#**********************************************************************************
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
env_graph = tf.Graph()
agn_graph = tf.Graph()
dis_graph = tf.Graph()
env_sess = tf.Session(config=config, graph=env_graph)
agn_sess = tf.Session(config=config, graph=agn_graph)
dis_sess = tf.Session(config=config, graph=dis_graph)

if FLAGS.use_simulated_data:
    data = load_data(FLAGS.data_dir, FLAGS.data_name)
else:
    data = load_data(FLAGS.data_dir, "train-sci-1c.csv")

#random_select = np.random.permutation(np.arange(len(data)))
#random_select = range(len(data))

# test_data = load_data(FLAGS.data_dir, "test_single.txt")
# single_data = []
# for i, s in enumerate(test_data):
#     tmps = []
#     for c in s:
#         c["click"] = aid2index[c["click"]] if c["click"] in aid2index else 1
#         c["rec_list"] = list(set([aid2index[rl] if rl in aid2index else 1 for rl in c["rec_list"]])) + [3]
#         if FLAGS.use_active_learning:
#             c["num"] = i
#         tmps.append(c)
#     single_data.append(tmps)
# print single_data

start_click = {}
for d in data:
    k = d[0]["click"]
    if k in start_click:
        start_click[k] += 1
    else:
        start_click[k] = 1
if len(start_click) < 2:
    sort_start_click = list(start_click.keys())
else:
    sort_start_click = sorted(start_click, key=start_click.get, reverse=True)[1:10000]

max_interact_len = min(6, 2 * int(np.mean([len(s) for s in data])))
print "average length:", np.mean([len(s) for s in data]), "max_interact_len:", max_interact_len
fold = len(data) / 40
data_train = data[:(fold * 38)]
data_dev = data[(fold * 38):(fold * 39)]
data_test = data[(fold * 39):]
if FLAGS.use_active_learning:
    data_train_active_num = list(np.zeros([len(data_train)]))
print "get training data:len(dataset) is %d " % (len(data_train))
print "get validation data:len(dataset) is %d " % (len(data_dev))
print "get testing data:len(dataset) is %d " % (len(data_test))

vocab, embed = build_vocab(data_train)
aid2index = {}
index2aid = {}
for i,a in enumerate(vocab):
    aid2index[a] = i
    index2aid[i] = a
def filter(d):
    if not FLAGS.use_simulated_data:
        new_d = []
        for i, s in enumerate(d):
            # unk = False
            # for c in s:
            #     if c["click"] not in aid2index:
            #         unk = True
            #         break
            # if unk:
            #     continue
            tmps = []
            for c in s:
                c["click"] = aid2index[c["click"]] if c["click"] in aid2index else 1
                c["rec_list"] = list(set([aid2index[rl] if rl in aid2index else 1 for rl in c["rec_list"]])) + [3]
                if FLAGS.use_active_learning:
                    c["num"] = i
                tmps.append(c)
            new_d.append(tmps)
        d = copy.deepcopy(new_d)
    return d
data_train = filter(data_train)
data_dev = filter(data_dev)
data_test = filter(data_test)

data_filter = []
for d in data_test:
    with_purchase = False
    if len(d) < 2:
        continue
    for c in d:
        if int(c["purchase"]) == 1:
            with_purchase = True
            break
    if with_purchase:
        data_filter.append(copy.deepcopy(d))
# with open("./data_filter.txt", "w") as fout:
#     for i,s in enumerate(data_filter):
#         for t in s:
#             print >> fout, str(i)+";;"+str(t["click"])+";"+",".join(map(str,t["rec_list"][:-1]))+";"+str(int(t["purchase"]))
# exit()
'''
with open("train_data.txt", "w") as fout:
    for k, s in enumerate(data_train):
        print >> fout, "------------next session:%d------------"%k
        for i in range(1, len(s)):
            print >> fout, "current_click:", s[i-1]["click"], "next_click:", s[i]["click"], "purchase:", s[i-1]["purchase"], "action:", s[i]["rec_list"]
exit()
'''

max_pool_size = int(6e5)
pool_size = int(6e4)
# max_pool_size = int(len(data_train))
# pool_size = int(len(data_train))

print np.mean([len(s) for s in data_train]), np.mean([len(s) for s in data_dev]), np.mean([len(s) for s in data_test])
if FLAGS.interact:
    with env_graph.as_default():
        env_model = EnvModel(
                len(embed),
                FLAGS.embed_units,
                FLAGS.units,
                FLAGS.layers,
                is_train=True,
                vocab=vocab,
                embed=embed)
        env_model.print_parameters()
        if tf.train.get_checkpoint_state(FLAGS.env_train_dir):
            #model_path = '%s/checkpoint-%08d' % (FLAGS.env_train_dir, 2041)
            #env_model.saver.restore(env_sess, model_path)
            print("Reading environment model parameters from %s" % FLAGS.env_train_dir)
            env_model.saver.restore(env_sess, tf.train.latest_checkpoint(FLAGS.env_train_dir))
        else:
            print("Created environment model with fresh parameters.")
            env_sess.run(tf.global_variables_initializer())

with agn_graph.as_default():
    agn_model = AgentModel(
            len(embed),
            FLAGS.embed_units,
            FLAGS.units,
            FLAGS.layers,
            is_train=True,
            embed=embed,
            action_num=FLAGS.action_num)
    agn_model.print_parameters()
    # agn_model.saver.restore(agn_sess, tf.train.latest_checkpoint(FLAGS.agn_train_dir))
    # reader = pywrap_tensorflow.NewCheckpointReader(FLAGS.agn_train_dir+"/checkpoint-00085089")
    # var_to_shape_map = reader.get_variable_to_shape_map()
    # # Print tensor name and values
    # for key in var_to_shape_map:
    #     print("tensor_name: ", key)
    if tf.train.get_checkpoint_state(FLAGS.agn_train_dir):
        print("Reading agent model parameters from %s" % FLAGS.agn_train_dir)
        agn_model.saver.restore(agn_sess, tf.train.latest_checkpoint(FLAGS.agn_train_dir))
    else:
        print("Created agent model with fresh parameters.")
        agn_sess.run(tf.global_variables_initializer())
if FLAGS.use_dis:
    with dis_graph.as_default():
        dis_model = DisModel(
                len(embed),
                FLAGS.embed_units,
                FLAGS.units,
                FLAGS.layers,
                is_train=True,
                vocab=vocab,
                embed=embed)
        dis_model.print_parameters()
        # agn_model.saver.restore(agn_sess, tf.train.latest_checkpoint(FLAGS.dis_train_dir))
        if tf.train.get_checkpoint_state(FLAGS.dis_train_dir):
            print("Reading dis model parameters from %s" % FLAGS.dis_train_dir)
            dis_model.saver.restore(dis_sess, tf.train.latest_checkpoint(FLAGS.dis_train_dir))
        else:
            print("Created dis model with fresh parameters.")
            dis_sess.run(tf.global_variables_initializer())
best_env_train_acc, best_env_train_acc_1, best_env_pg_train_acc, best_env_pg_train_acc_1, best_agn_train_acc, best_agn_train_acc_1 = 0., 0., 0., 0., 0., 0.
def env_train(size):
    global best_env_train_acc, best_env_train_acc_1
    pre_losses = [1e18] * 3
    for _ in range(size):
        with env_graph.as_default():
            epoch = env_model.epoch.eval(session=env_sess)
            start_time = time.time()
            loss, pr_loss, pu_loss, _, _ = env_model.train(env_sess, data_train)
            if loss > max(pre_losses):  # Learning rate decay
                env_sess.run(env_model.learning_rate_decay_op)
            pre_losses = pre_losses[1:] + [loss]
            print "epoch %d lr %.4f time %.4f ppl [%.8f] pr_loss [%.8f] pu_loss [%.8f]" \
                  % (epoch, env_model.learning_rate.eval(session=env_sess), time.time() - start_time, loss, pr_loss, pu_loss)
            loss, pr_loss, pu_loss, acc, acc_1 = env_model.train(env_sess, data_dev, is_train=False)
            print "        dev_set, ppl [%.8f] pr_loss [%.8f] pu_loss [%.8f] best_p@%d [%.4f]" % (loss, pr_loss, pu_loss, FLAGS.metric, best_env_train_acc)
            if acc > best_env_train_acc or acc_1 > best_env_train_acc_1:
                if acc > best_env_train_acc: best_env_train_acc = acc
                if acc_1 > best_env_train_acc_1: best_env_train_acc_1 = acc_1
                loss, pr_loss, pu_loss, _, _ = env_model.train(env_sess, data_test, is_train=False)
                print "        test_set, ppl [%.8f] pr_loss [%.8f] pu_loss [%.8f] best_p@%d [%.4f]" % (loss, pr_loss, pu_loss, FLAGS.metric, best_env_train_acc)
                env_model.saver.save(env_sess, '%s/checkpoint' % FLAGS.env_train_dir, global_step=env_model.global_step)
                print("Saving env model params in %s" % FLAGS.env_train_dir)
            print "------env train finish-------"

def env_pg_train(size):
    global best_env_pg_train_acc, best_env_pg_train_acc_1
    pre_losses = [1e18] * 3
    for _ in range(size):
        with env_graph.as_default():
            epoch = env_model.epoch.eval(session=env_sess)
            start_time = time.time()
            loss = env_model.pg_train(env_sess, generate_session[-pool_size:])
            if loss > max(pre_losses):  # Learning rate decay
                env_sess.run(env_model.learning_rate_decay_op)
            pre_losses = pre_losses[1:] + [loss]
            print "epoch %d lr %.4f time %.4f ppl [%.8f]" \
                  % (epoch, env_model.learning_rate.eval(session=env_sess), time.time() - start_time, loss)
            loss, pr_loss, pu_loss, acc, acc_1 = env_model.train(env_sess, data_dev, is_train=False)
            print "        dev_set, ppl [%.8f] pr_loss [%.8f] pu_loss [%.8f]" % (loss, pr_loss, pu_loss)
            if acc > best_env_pg_train_acc or acc_1 > best_env_pg_train_acc_1:
                if acc > best_env_pg_train_acc: best_env_pg_train_acc = acc
                if acc_1 > best_env_pg_train_acc_1: best_env_pg_train_acc_1 = acc_1
                loss, pr_loss, pu_loss, _, _ = env_model.train(env_sess, data_test, is_train=False)
                print "        test_set, ppl [%.8f] pr_loss [%.8f] pu_loss [%.8f]" % (loss, pr_loss, pu_loss)
                env_model.saver.save(env_sess, '%s/checkpoint' % FLAGS.env_train_dir, global_step=env_model.global_step)
                print("Saving env model params in %s via pg" % FLAGS.env_train_dir)
            print "------env pg train finish-------"

def agn_train(size):
    global data_train_active_num, best_agn_train_acc, best_agn_train_acc_1
    for _ in range(size):
        with agn_graph.as_default():
            epoch = agn_model.epoch.eval(session=agn_sess)
            #random.shuffle(data_train)
            start_time = time.time()
            if FLAGS.use_active_learning:
                loss, acc, acc_1, data_train_active_num = agn_model.train_active(agn_sess, data_train, generate_session[-pool_size:], data_train_active_num)
                with open("./active_record.txt", "w") as fout:
                    for n in data_train_active_num:
                        print >> fout, n
            else:
                loss, acc, acc_1 = agn_model.train(agn_sess, data_train, generate_session[-pool_size:])
            #if loss > max(pre_losses):  # Learning rate decay
                #agn_sess.run(agn_model.learning_rate_decay_op)
            #pre_losses = pre_losses[1:] + [loss]
            print "epoch %d learning rate %.4f epoch-time %.4f perplexity [%.8f] p@%d %.4f%% p@1 %.4f%%" \
                    % (epoch, agn_model.learning_rate.eval(session=agn_sess), time.time() - start_time, loss, FLAGS.metric, acc*100, acc_1*100)
            loss, acc, acc_1 = agn_model.train(agn_sess, data_dev, is_train=False)
            print "        dev_set, perplexity [%.8f] p@%d %.4f%% p@1 %.4f%%" % (loss, FLAGS.metric, acc*100, acc_1*100)
            if acc > best_agn_train_acc or acc_1 > best_agn_train_acc_1:
                if acc > best_agn_train_acc: best_agn_train_acc = acc
                if acc_1 > best_agn_train_acc_1: best_agn_train_acc_1 = acc_1
                loss, acc, acc_1 = agn_model.train(agn_sess, data_test, is_train=False)
                print "        test_set, perplexity [%.8f] p@%d %.4f%% p@1 %.4f%%" % (loss, FLAGS.metric, acc*100, acc_1*100)
                agn_model.saver.save(agn_sess, '%s/checkpoint' % FLAGS.agn_train_dir, global_step=agn_model.global_step)
                print("Saving agn model params in %s" % FLAGS.agn_train_dir)
            print "------agn train finish-------"

def dis_train(size):
    random_generate_session = list(np.random.permutation(generate_session))
    data_gen_train = random_generate_session[:len(generate_session)*19/20]
    data_gen_test = random_generate_session[-len(generate_session)/20:]
    for _ in range(size):
        with dis_graph.as_default():
            dis_epoch = dis_model.epoch.eval(session=dis_sess)
            #random.shuffle(data_train)
            start_time = time.time()

            loss, acc = dis_model.train(data_train, data_gen_train, sess=dis_sess)
            dis_model.saver.save(dis_sess, '%s/checkpoint' % FLAGS.dis_train_dir, global_step=dis_model.global_step)
            print "dis_epoch %d learning rate %.4f epoch-time %.4f perplexity [%.8f] acc %.4f%%" \
                    % (dis_epoch, dis_model.learning_rate.eval(session=dis_sess), time.time() - start_time, loss, acc*100)
            loss, acc = dis_model.train(data_dev, data_gen_test, sess=dis_sess, is_train=False)
            print "        test_set, perplexity [%.8f] acc %.4f%% " % (loss, acc*100)

            print "------dis train finish-------"

def interact(size, num=1000, use_dis=FLAGS.use_dis):
    start_time = time.time()
    for _ in range(size):
        flog = open("%s/train_log_%d.txt"%(FLAGS.interact_data_dir, session_no), "w")
        generate_data(min(num, pool_size, len(data)), flog, use_dis=use_dis)
        flog.close()
    print "generate-time %.4f" % (time.time()-start_time)

# env_train(50)
# exit()
# agn_train(50)
# exit()

pre_losses = [1e18] * 3
#pre-train the discriminator
if FLAGS.interact:
    interact(1, use_dis=False)
    if FLAGS.use_dis:
        dis_train(3)
generate_session = []
ua_step = 3
a_step = 3
# while True: 
for _ in range(50):
    for _ in range(ua_step):
        for _ in range(a_step):
            if FLAGS.interact:
                interact(1)
            agn_train(1)

        if FLAGS.use_dis:
            env_pg_train(1)
            env_train(1)
    if FLAGS.use_dis:
        dis_train(3)
    print "*"*25
    # with env_graph.as_default():
        # env_model.saver.restore(env_sess, tf.train.latest_checkpoint(FLAGS.env_train_dir))
with open("./generate_session.txt", "w") as fout:
    # key = ["session_no", "click", "rec_list", "purchase"]
    # print >> fout, " ".join(key)
    for s in generate_session:
        for t in s:
            print >> fout, str(t["session_no"])+";;"+str(t["click"])+";"+",".join(map(str,t["rec_list"][:-1]))+";"+str(int(t["purchase"]))
