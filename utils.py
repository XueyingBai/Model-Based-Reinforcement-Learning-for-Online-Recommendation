import tensorflow as tf
import numpy as np
import os
import copy
tf.app.flags.DEFINE_boolean("is_train", True, "Set to False to inference.")
tf.app.flags.DEFINE_integer("units", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("symbols", 40000, "Size of symbol list.")
tf.app.flags.DEFINE_integer("embed_units", 50, "Embedding units.")
tf.app.flags.DEFINE_integer("layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("action_num", 10, "num of recommendations")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size to use during training.")

# simulation_train  simulation_ckp_deter  simulation_ckp_half2half
tf.app.flags.DEFINE_string("data_dir", "../data_session/", "Data directory")
# tf.app.flags.DEFINE_string("data_dir", "../simulator/interact_data_mix_s10_a50/", "Data directory")
tf.app.flags.DEFINE_string("data_name", "random_generate_session_s10_a50.txt", "Data directory")
tf.app.flags.DEFINE_integer("data_size", 10000, "") # random deterministic half2half
tf.app.flags.DEFINE_float("random_prob", 0.0, "") # random deterministic half2half
tf.app.flags.DEFINE_string("interact_data_dir", "./simulation_train_mix_s10_a50/random_prob/interact_data_env_agn", "Training directory.")
tf.app.flags.DEFINE_string("agn_train_dir", "./simulation_train_mix_s10_a50/random_prob/agn_pre_train", "Training directory.")
tf.app.flags.DEFINE_string("env_train_dir", "./simulation_train_mix_s10_a50/random_prob/env_pre_train", "Training directory.")
tf.app.flags.DEFINE_string("dis_train_dir", "./simulation_train_mix_s10_a50/random_prob/dis_train", "Training directory.")
tf.app.flags.DEFINE_boolean("use_dis", True, "Set to True to show the parameters")
tf.app.flags.DEFINE_boolean("interact", True, "Set to True to show the parameters")
tf.app.flags.DEFINE_boolean("use_simulated_data", False, "Set to True to use simulated data")
tf.app.flags.DEFINE_integer("metric", 3, "compute p@metric")

tf.app.flags.DEFINE_boolean("use_is", False, "Set to True to use importance sampling")
tf.app.flags.DEFINE_boolean("use_active_learning", False, "Set to True to use active learning")
tf.app.flags.DEFINE_integer("per_checkpoint", 1000, "How many steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("inference_version", 36582, "The version for inferencing.")
tf.app.flags.DEFINE_boolean("log_parameters", True, "Set to True to show the parameters")
tf.app.flags.DEFINE_string("inference_path", "", "Set filename of inference, default isscreen")
tf.app.flags.DEFINE_float("min_epsilon", 1e-4, "when epsilon < this num, not random choose any longer")
tf.app.flags.DEFINE_float("max_epsilon", 0.1, "initial epsilon")
tf.app.flags.DEFINE_float("gamma", 0.9, "discount factor")
tf.app.flags.DEFINE_integer("OBSERVE", 1e6, "the num of steps before training")
tf.app.flags.DEFINE_integer("memory_size", 5000, "memory upper limit")

FLAGS = tf.app.flags.FLAGS

PAD_ID = 0
UNK_ID = 1
GO_ID = 2
EOS_ID = 3
_START_VOCAB = ['_PAD', '_UNK', '_GO', '_EOS']

def sigmoid(x):
    return 1 / (1+ np.exp(-x))

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

def load_data(path, fname):
    session = {}
    output_session = []
    print("Reading data from:", path+fname)
    with open(path + fname, "r") as fin:
        #fin.readline()
        for line in fin:
            tmp = line.strip().split(";")
            if int(tmp[0]) in session:
                session[int(tmp[0])].append(
                    {"click": int(tmp[2]) if FLAGS.use_simulated_data else tmp[2],
                    "rec_list": map(int, tmp[3].strip().split(",")) if FLAGS.use_simulated_data else tmp[3].strip().split(","),
                    "purchase": 0 if FLAGS.use_simulated_data else float(tmp[4].strip()),
                    "dis_reward":1.})
            else:
                session[int(tmp[0])] = [
                    {"click": int(tmp[2]) if FLAGS.use_simulated_data else tmp[2],
                    "rec_list": [] if FLAGS.use_simulated_data else tmp[3].strip().split(","),
                    "purchase": 0 if FLAGS.use_simulated_data else float(tmp[4].strip()),
                    "dis_reward":1.}]
            if session[int(tmp[0])][-1]["click"] not in session[int(tmp[0])][-1]["rec_list"]:
                session[int(tmp[0])][-1]["rec_list"] += [session[int(tmp[0])][-1]["click"]]
            if session[int(tmp[0])][-1]["click"] not in session[int(tmp[0])][-1]["rec_list"]:
                session[int(tmp[0])][-1]["rec_list"] += [session[int(tmp[0])][-1]["click"]]

        skey = sorted(session.keys())

        for key in skey:
            # if len(session[key]) > 1:
            if len(session[key]) > 1 and len(session[key]) <= 40:
                output_session.append(session[key])
        print len(output_session) #(1,40)118,860    total 119,366
    return output_session

# def load_simulation_data(path):
#     session = {}
#     output_session = []
#     all_reward = []
#     '''
#     with open(path + "gen_click.txt", "r") as fclick:
#         with open(path + "gen_reward.txt", "r") as freward:
#             for line_click in fclick:
#                 tmp_click = line_click.strip().split()
                
#                 line_reward = freward.readline()
#                 tmp_reward = map(float, line_reward.strip().split())

#                 tmp_output_session = []
#                 for c, r in zip(tmp_click, tmp_reward):
#                     tmp_output_session.append({"click":c, "rec_list":["%d"%i for i in range(19)], "purchase": (r-1) / 3., "dis_reward":1.})
#                     all_reward.append(r)
#                 output_session.append(tmp_output_session)
#         print len(output_session) #(1,40)118,860    total 119,366
#         print "mean:",np.mean(all_reward), "\tvariance:",np.var(all_reward)
#     '''
#     with open(path + "gen_data.txt", "r") as fclick:
#         for line_click in fclick:
#             tmp_click = line_click.strip().split()
            
#             tmp_output_session = []
#             for c in tmp_click:
#                 tmp_output_session.append({"click":c, "rec_list":["%d"%i for i in range(19)], "purchase": (r-1) / 3., "dis_reward":1.})
#                 all_reward.append(r)
#             output_session.append(tmp_output_session)
#         print len(output_session) #(1,40)118,860    total 119,366
#         print "mean:",np.mean(all_reward), "\tvariance:",np.var(all_reward)
#     return output_session


def build_vocab(data):
    print("Creating vocabulary...")
    if FLAGS.use_simulated_data == True:
        print "Reading from article list from %s ......" % FLAGS.data_dir
        article_list = copy.deepcopy(_START_VOCAB)
        with open("%s/article_list_a50.txt"%FLAGS.data_dir, "r") as fin:
            for line in fin:
                tmp = line.strip().split()
                article_list.append(tmp[0])
        if '_PAD' not in article_list:
            article_list = _START_VOCAB + article_list
    else:
        vocab = {}
        for each_session in data:
            for item in each_session:
                if FLAGS.use_simulated_data:
                    v = [item["click"]] + item["rec_list"]
                else:
                    v = [item["click"]]
                for token in v:
                    if token in vocab:
                        vocab[token] += 1
                    else:
                        vocab[token] = 1
        article_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)

    with open("article_list.txt", "w") as fout:
        for a in article_list:
            print >> fout, a
    if len(article_list) > FLAGS.symbols:
        article_list = article_list[:FLAGS.symbols]
    else:
        FLAGS.symbols = len(article_list)
    embed = []
    for i, _ in enumerate(article_list):
        if i < len(_START_VOCAB):
            embed.append(np.zeros(FLAGS.embed_units, dtype=np.float32))
        else:
            embed.append(np.random.normal(size=(FLAGS.embed_units)))
    embed = np.array(embed, dtype=np.float32)
    return article_list, embed


def gen_batched_data(data):
    random_appendix = 100
    max_len = max([len(item) for item in data]) + 1
    max_rec_len = max([len(s["rec_list"]) for item in data for s in item] + [random_appendix+20])
    sessions, aims, rec_lists, sessions_length, purchase, rec_mask,  dis_reward, env_dis_reward, cum_env_dis_reward, cum_env_reward = [], [], [], [], [], [], [], [], [], []
    def padding(sent, l):
        return sent + [3] + [0 for _ in range(l-len(sent)-1)]

    def padding_m1(vec, l):
        return vec + [-1. for _ in range(l - len(vec))]

    def padding_cum_reward(vec, l):
        for i in range(len(vec)-1):
            inv_i = len(vec) - i - 2
            vec[inv_i] += FLAGS.gamma * vec[inv_i+1]
        return vec + [-1. for _ in range(l - len(vec))]

    def get_vec(session_al):
        session_tmp = []
        mask_tmp = []
        for al in session_al:
            session_tmp.append(al + [0 for _ in range(max_rec_len - len(al))])
            mask_tmp.append([1. for _ in range(len(al))] + [0. for _ in range(max_rec_len - len(al))])
        session_tmp += [[0 for _ in range(max_rec_len)] for k in range(max_len-len(session_tmp))]
        mask_tmp += [[0. for _ in range(max_rec_len)] for  k in range(max_len-len(mask_tmp))]
        return session_tmp, mask_tmp

    def get_aim(session_aim, rec_list):
        s_aim = []
        for a, rlist in zip(session_aim, rec_list):
            try:
                s_aim.append(rlist.index(a))
            except ValueError:
                s_aim.append(len(rlist))
                #print "ValueError", a, rlist
        return s_aim

    for item in data:
        sessions.append(padding([s["click"] for s in item], max_len))    
        purchase.append(padding_m1([s["purchase"] for s in item][1:] + [0.], max_len))

        env_reward = np.array([s["purchase"] * 3 + 1 for s in item][1:] + [0.])
        dis_reward.append(padding_m1([s["dis_reward"] for s in item], max_len))
        env_dis_reward.append(padding_m1(list(env_reward*np.array([s["dis_reward"] for s in item])), max_len))
        cum_env_reward.append(padding_cum_reward(list(env_reward), max_len))
        cum_env_dis_reward.append(padding_cum_reward(list(env_reward*np.array([s["dis_reward"] for s in item])), max_len))

        rl, rm = get_vec([s["rec_list"] for s in item][1:] + [[3]+list(np.random.permutation(range(4,FLAGS.symbols))[:np.random.randint(15,random_appendix)])])
        rec_lists.append(rl)
        rec_mask.append(rm)
        aims.append(get_aim(sessions[-1][1:] + [0], rl))
        sessions_length.append(len(item)+1)

    batched_data = {'sessions': np.array(sessions), # the click sequence [batch_size, encoder_length]
            'rec_lists': np.array(rec_lists),   # the recommendation sequence [batch_size, encoder_length, rec_length]
            'rec_mask' : np.array(rec_mask),    # the mask of recommendation sequence [batch_size, encoder_length, rec_length]
            'aims': np.array(aims), # aim index [batch_size, encoder_length]
            'purchase': np.array(purchase), # 0. or 1. to indicate whether purchase the next click [batch_size, encoder_length]
            'cum_env_reward': np.array(cum_env_reward),    # cum purchase reward from env
            'dis_reward': np.array(dis_reward), # reward \in [0,1] from dis
            'env_dis_reward': np.array(env_dis_reward),    # purchase reward \in [0, 4] from env reweighted by dis
            'cum_env_dis_reward': np.array(cum_env_dis_reward), # cum reward from env reweighted by dis
            'sessions_length': sessions_length} # session_length
    # np.set_printoptions(threshold=np.inf)
    # for key in batched_data:
    #     print key, np.shape(batched_data[key]), batched_data[key][:2]
    # exit()
    return batched_data


def compute_acc(ba, pi, rl, mask, purchase, ftest_name, output):
    ftest = open(ftest_name, "a+")
    total_num, total_num_1, correct, correct_1 = 0., 0., 0., 0.
    pur, pur_1, all_purchase, all_purchase_1 = 0., 0., 0., 0.
    for batch_aim, batch_predict_idx, batch_rec_list, batch_rec_mask, batch_purchase in zip(ba, pi, rl, mask, purchase):
        for i, (aim, predict_index, rec_list, rec_mask, purchase) in enumerate(zip(batch_aim, batch_predict_idx, batch_rec_list, batch_rec_mask, batch_purchase)):
            if rec_list[0] == 3:
                break
            if np.sum(rec_mask) > FLAGS.metric:
                if output: print >> ftest, ">%d"%FLAGS.metric,
                total_num += 1
                for tmpp in predict_index:
                    if rec_list[tmpp] != 1 and rec_mask[tmpp] == 1 and (aim == tmpp):
                        if output: print >> ftest, "p@%d"%FLAGS.metric,
                        correct += 1
                        break

                if batch_purchase[i+1] == 1:
                    all_purchase += 1
                    for tmpp in predict_index:
                        if rec_list[tmpp] != 1 and rec_mask[tmpp] == 1 and (aim == tmpp):
                            pur += 1
                            break
            if np.sum(rec_mask) > 1:
                total_num_1 += 1
                if rec_list[predict_index[0]] != 1 and rec_mask[predict_index[0]] == 1 and (aim == predict_index[0]):
                    if output: print >> ftest, 1,
                    correct_1 += 1
                else:
                    if output: print >> ftest, 0,
                if output: print >> ftest, batch_purchase[i+1] * 3 + 1,
                try:
                    if output: print >> ftest, [rec_list[tmpp] for tmpp in predict_index], rec_list[aim], list(rec_list)
                except:
                    print predict_index, aim, rec_list
                    print ba, pi, rl, mask, purchase
                    exit()
                if batch_purchase[i+1] == 1:
                    all_purchase_1 += 1
                    if rec_list[predict_index[0]] != 1 and rec_mask[predict_index[0]] == 1 and (aim == predict_index[0]):
                        pur_1 += 1
            else:
                if output: print >> ftest, "_1_", [rec_list[tmpp] for tmpp in predict_index], rec_list[aim], list(rec_list) 

        if output: print >> ftest, "-------------------------"
    return correct / (total_num + 1e-18), correct_1 / (total_num_1 + 1e-18), pur, all_purchase, pur_1, all_purchase_1
