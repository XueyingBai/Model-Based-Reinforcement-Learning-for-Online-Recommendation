from argparse import ArgumentParser
def get_args():
    parser = ArgumentParser(description='Time_Series_Item_Prediction')
    parser.add_argument('--data', type=str, default='../TimeSeries/data_session',
                        help='location of the directory of the training data')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_Tanh, RNN_RELU, LSTM, GRU)')
    parser.add_argument('--embed_dim', type=int, default=50,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=512,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='number of layers')
    parser.add_argument('--pre_window', type=int, default=0.5, 
                        help='the propotion of the given sentence length')
    parser.add_argument('--dropout', type=float, default=0,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed for reproducibility')
    parser.add_argument('--feature_vector', type=str, default='../TimeSeries/itemTransformedFeature_2',
                        help='Path of feature_vectors')    
    parser.add_argument('--window', type=float, default=0.8,
                        help='Window Size')
    parser.add_argument('--init_embed', action='store_true',
                    help='whether to initialize the word embedding')
    args = parser.parse_args()
    return args