import argparse

def __parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str,
                        help='train|test')
    parser.add_argument('--model', default='obj', type=str,
                        help='obj|act|both')
    parser.add_argument('--model_path1', default='', type=str,
                        help='path to resnet model for test')
    parser.add_argument('--model_path2', default='', type=str,
                        help='path to i3d audio model for test')
    parser.add_argument('--cpv', default=20, type=int,
                        help='the number of captions per video')
    parser.add_argument('--data_path', default='../../data',
                        help='path to datasets')
    parser.add_argument('--cap_train_path', default='train.npy',
                        help='path to npy file of caption of train set')
    parser.add_argument('--cap_val_path', default='val.npy',
                        help='path to npy file of caption of validation set')
    parser.add_argument('--cap_test_path', default='test.npy',
                        help='path to npy file of caption of test set')
    parser.add_argument('--resnet_path', default='resnet',
                        help='resnet feature dir relative to data_path')
    parser.add_argument('--rgbi3d_path', default='rgb_i3d',
                        help='i3d feature dir relative to data_path')
    parser.add_argument('--soundnet_path', default='soundnet',
                        help='soundnet feature dir relative to data_path')
    parser.add_argument('--flowi3d_path', default='flow_i3d',
                        help='optical flow I3D feature dir relative to data_path')
    parser.add_argument('--finetune', default=False, type=bool,
                        help='fine tune the the pre-train model. NOT AVAILABLE YET')
    parser.add_argument('--vocab_path', default='../../data/glove.twitter.100d.pkl',
                        help='Path to saved vocabulary pickle files.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--weight_decay', default=0, type=float,
                        help='weight decay loss.')
    parser.add_argument('--num_epochs', default=40, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=100, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--word2vec_path', default='../../data/glove.twitter.100d.npy', type=str,
                        help='The path of weights of word embeding in npy format')
    parser.add_argument('--word2vec_trainable', action='store_true',
                        help='Finetune the word embeding layer')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=8, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=2, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=50000, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--log_path', default='runs/runX',
                        help='Path to save the model and Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--input_dim', default=2048, type=int,
                        help='Dimensionality of the input. 2048 or 3072.')
    parser.add_argument('--measure', default='cosine',
                        help='Similarity measure used (cosine|order)')
    parser.add_argument('--use_abs', action='store_true',
                        help='Take the absolute value of embedding vectors.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    return parser.parse_args()

CONSTANT = __parse_args()