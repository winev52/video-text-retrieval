import numpy as np
import os.path as path
import pickle
import nltk
import argparse

from vocabulary import Vocabulary

def get_indexed_caps(data, vocab):
    # transformation function
    def _transform_cap_record(rec):
        video_id, cap = rec
        tokens = nltk.tokenize.casual_tokenize(str(cap).lower())
        
        encoded = [word2idx[t] if t in word2idx else unk_idx for t in tokens]
        # encoded = [vocab.word2idx['<start>'], *encoded, vocab.word2idx['<end>']]

        return video_id, np.array(encoded, dtype=np.int64)

    # dict or Vocabulary
    if isinstance(vocab, dict):
        word2idx = vocab
        unk_idx = word2idx['<unknown>']
    else:
        word2idx = vocab.word2idx
        unk_idx = word2idx['<unk>']
    # do transform
    return np.array(list(map(_transform_cap_record, data)), dtype=object)

def create_split(data, split, cpv):
    n_instances = len(data)
    n_videos = n_instances // cpv
    total = np.sum(split)

    split_points = [cpv*int(x*n_videos/total) for x in split[:-1]]
    split_data = np.split(data, np.cumsum(split_points))

    return split_data


def _create_files(split, in_file, vocab_file, output_dir, cpv):
    # load data
    with open(in_file, 'rb') as f:
        data = pickle.load(f)

    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)

    # shuffle and then sort by id
    np.random.shuffle(data)
    data = data[np.argsort(data[:, 0])]
    # group by id
    vid_ids, counts = np.unique(data[:,0], return_counts=True)
    vid_list = np.split(np.arange(len(data)), np.cumsum(counts))
    # select cpv captiopn per video
    selected_index = [clist[:cpv] for clist in vid_list if len(clist) >= cpv]
    selected_index = np.concatenate(selected_index)
    data = data[selected_index]

    # transform to index
    data = get_indexed_caps(data, vocab)

    # create split
    train, val, test = create_split(data, split, cpv)

    np.save(path.join(output_dir, 'train.npy'), train)
    np.save(path.join(output_dir, 'val.npy'), val)
    np.save(path.join(output_dir, 'test.npy'), test)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=1200, type=float,
                        help='portion of train data')
    parser.add_argument('--val', default=100, type=float,
                        help='portion of validation data')
    parser.add_argument('--test', default=670, type=float,
                        help='portion of test data')
    parser.add_argument('--input', default='../../data/msvd_video_caps.pkl', type=str,
                        help='path to the output')
    parser.add_argument('--vocab', default='./vocab/vocab.pkl', type=str,
                        help='path to the output')
    parser.add_argument('--output', default='../../data', type=str,
                        help='path to the output')
    parser.add_argument('--cpv', default=20, type=int,
                        help='captions per video')

    opt = parser.parse_args()
    print(opt)
    _create_files([opt.train, opt.val, opt.test], opt.input, opt.vocab, opt.output, opt.cpv)

    return 0

if __name__ == "__main__":
    main()