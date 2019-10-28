from argparse import ArgumentParser
import inspect
from torch import optim
import re
def get_args():
    parser = ArgumentParser(description='Time_Series_Item_Prediction')
    parser.add_argument('--click', type=str, default='../dig_seq_click_rl.txt',
                        help='location of the directory of the click data')
    parser.add_argument('--purchase', type=str, default='../dig_seq_purchase_rl.txt',
                        help='location of the directory of the purchase data')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_Tanh, RNN_RELU, LSTM, GRU)')
    parser.add_argument('--embed_dim', type=int, default=50,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=50,
                        help='number of hidden units per layer')
    parser.add_argument('--embed_dim_policy', type=int, default=50,
                        help='size of word embeddings for policy')
    parser.add_argument('--nhid_policy', type=int, default=50,
                        help='number of hidden units per layer for policy')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='number of layers')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='batch size')
    parser.add_argument('--dropout', type=float, default=0,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed for reproducibility')
    parser.add_argument('--feature_vector', type=str, default='./itemTransformedFeature_2',
                        help='Path of feature_vectors')    
    parser.add_argument('--window', type=float, default=0.8,
                        help='Window Size')
    parser.add_argument('--correlated_id', type=str, default='./itemIDTransform_session',
                        help='location of the correlated_id list')
    parser.add_argument('--init_embed', action='store_true',
                    help='whether to initialize the word embedding')
    parser.add_argument('--optim', type=str, default='sgd,lr=0.1',
                        help='type of the optimizer')
    parser.add_argument('--load', action = 'store_true',
                        help='load existed environment and policy')
    args = parser.parse_args()
    return args

def get_optimizer(s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))
    return optim_fn, optim_params