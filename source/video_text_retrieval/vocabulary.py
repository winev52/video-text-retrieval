
class Vocabulary:
    def __init__(self, word2idx, idx2word):
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.idx = len(idx2word)

    def __len__(self):
        return self.idx