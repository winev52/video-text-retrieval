import numpy as np
import os.path as path
import pickle
import nltk
import argparse

from vocabulary import Vocabulary

def get_indexed_caps(caps_pkl_path, vocab_pkl_path):
    # load data
    with open(vocab_pkl_path, 'rb') as f:
        vocab = pickle.load(f)

    with open(caps_pkl_path, 'rb') as f:
        data = pickle.load(f)

    # transformation function
    def _transform_cap_record(rec):
        video_id, cap = rec
        tokens = nltk.tokenize.casual_tokenize(str(cap).lower())
        encoded = [vocab.word2idx[t] if t in vocab.word2idx else vocab.word2idx['<unk>'] for t in tokens]
        encoded = [vocab.word2idx['<start>'], *encoded, vocab.word2idx['<end>']]
        
        return video_id, np.array(encoded)

    # do transform
    return np.array(list(map(_transform_cap_record, data)), dtype=object)

def create_train_val_test(data, split):
    n_instances = len(data)
    train, val, test = split
    total = train + val + test

    # calc the number of samples for each set
    train = int(n_instances * train / total)
    val = int(n_instances * val / total)
    test = n_instances - train - val

    # shuffle data
    np.random.shuffle(data)
    return data[:train], data[train: train + val], data[-test:]

def _create_files(split, input, vocab, output):
    data = get_indexed_caps(input, vocab)
    train, val, test = create_train_val_test(data, split)

    np.save(path.join(output, 'train.npy'), train)
    np.save(path.join(output, 'val.npy'), val)
    np.save(path.join(output, 'test.npy'), test)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=0.85, type=float,
                        help='portion of train data')
    parser.add_argument('--val', default=0.05, type=float,
                        help='portion of validation data')
    parser.add_argument('--test', default=0.10, type=float,
                        help='portion of test data')
    parser.add_argument('--input', default='../../data/msvd_video_caps.pkl', type=str,
                        help='path to the output')
    parser.add_argument('--vocab', default='./vocab/vocab.pkl', type=str,
                        help='path to the output')
    parser.add_argument('--output', default='../../data', type=str,
                        help='path to the output')

    opt = parser.parse_args()
    print(opt)
    _create_files([opt.train, opt.val, opt.test], opt.input, opt.vocab, opt.output)

    return 0

if __name__ == "__main__":
    main()