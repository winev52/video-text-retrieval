import torch
import torch.nn as nn
import torch.nn.init
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
import sys


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1).sqrt()
    X = X / norm[:,None]
    return X


# We consider Image feature is precomputed
class EncoderImage(nn.Module):

    def __init__(self, input_dim, embed_size, use_abs=False, no_imgnorm=False, dropout=0.5):
        super(EncoderImage, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        if dropout != 0:
            self.dropout = nn.Dropout(p=0.25)

        self.fc1 = nn.Linear(input_dim, embed_size)
        # self.a1 = nn.LeakyReLU()
        # self.fc2 = nn.Linear(embed_size*2, embed_size)

        # self.init_weights()

    # def init_weights(self):
    #     """Xavier initialization for the fully connected layer
    #     """
    #     pass
        # r = np.sqrt(6.) / np.sqrt(self.fc1.in_features +
        #                           self.fc1.out_features)
        # self.fc1.weight.data.uniform_(-r, r)
        # self.fc1.bias.data.fill_(0)

        # r = np.sqrt(6.) / np.sqrt(self.fc2.in_features +
        #                           self.fc2.out_features)

        # self.fc2.weight.data.uniform_(-r, r)
        # self.fc2.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        if hasattr(self, 'dropout'):
            features = self.dropout(images)
        features = self.fc1(features)
        # features = self.a1(features)
        # features = self.fc2(features)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features

    # def load_state_dict(self, state_dict):
    #     """Copies parameters. overwritting the default one to
    #     accept state_dict from Full model
    #     """
    #     own_state = self.state_dict()
    #     new_state = OrderedDict()
    #     for name, param in state_dict.items():
    #         if name in own_state:
    #             new_state[name] = param

    #     super(EncoderImage, self).load_state_dict(new_state)



# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_abs=False,  no_imgnorm=False):
        super(EncoderText, self).__init__()
        self.use_abs = use_abs
        self.no_imgnorm = no_imgnorm
        self.embed_size = embed_size

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True)

        # self.fc = nn.Linear(embed_size*2, embed_size)

        # self.init_weights()

    # def init_weights(self):
    #     pass
        # self.embed.weight.data.uniform_(-0.1, 0.1)

        # r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
        #                           self.fc.out_features)
        # self.fc.weight.data.uniform_(-r, r)
        # self.fc.bias.data.fill_(0)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        
        packed = pack_padded_sequence(x, lengths, batch_first=True)
        # print(np.sum(lengths))
        # print(packed.data.size())
        # Forward propagate RNN
        out, _ = self.rnn(packed)
        # print(out.data.size())

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        I = torch.LongTensor(lengths).view(-1, 1, 1)
        I = Variable(I.expand(x.size(0), 1, self.embed_size)-1).cuda()
        # print(padded[0][24][lengths[24] - 1])
        out = torch.gather(padded[0], 1, I).squeeze(1)
        # print(out[24].data.cpu().numpy())
        # out = self.fc(out)
        # print(list(self.rnn.parameters()))

        # normalization in the joint embedding space
        if not self.no_imgnorm:
            out = l2norm(out)

        # take absolute value, used by order embeddings
        if self.use_abs:
            out = torch.abs(out)

        return out


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    # nim = torch.pow(im, 2).sum(dim=1).sqrt().view(im.size(0), 1)
    # print(nim.shape)
    # ns = torch.pow(s, 2).sum(dim=1).sqrt().view(1, s.size(0))
    # print(ns.shape)
    # dom = nim.mm(ns)
    # return torch.div(im.mm(s.t()), dom)
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).squeeze(2).sqrt().t()
    return score


class Loss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(Loss, self).__init__()
        self.margin = margin
        self.sim = cosine_sim
        self.max_violation = max_violation

    def forward(self, im, s):
        # print('im', im[:3, :10].data.cpu().numpy())
        # print('s', s[:3, :10].data.cpu().numpy())
        # sys.exit(0)
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        # print(diagonal)
        # sys.exit(0)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # print('score', scores[0].data.cpu().numpy())

        d1_sort, d1_indice=torch.sort(scores)
        # print(d1_sort[0])
        # sys.exit(0)
        rank_weights1 = Variable(torch.zeros(scores.size(0)).cuda())
        d1_indice_data = d1_indice.data.cpu().numpy()
        for i in range(d1.size(0)):
            for j in range(d1_indice.size(1)):
                if d1_indice_data[i][j] == i:
                    break
            rank_weights1[i] = 1/(j + 1)
        
        # d1_sort, d1_indice=torch.sort(scores)
        # val, id1 = torch.min(d1_indice,1)
        # rank_weights1 = id1.float()
        
        # for j in range(d1.size(0)):
        #     rank_weights1[j]=1/(rank_weights1[j]+1)

        d2_sort, d2_indice=torch.sort(scores.t())
        rank_weights2 = Variable(torch.zeros(scores.size(0)).cuda())
        d2_indice_data = d2_indice.data.cpu().numpy()
        for i in range(d2.size(0)):
            for j in range(d2_indice.size(1)):
                if d2_indice_data[i][j] == i:
                    break
            rank_weights2[i] = 1/(j + 1)
        
        # d2_sort, d2_indice=torch.sort(scores.t())
        # val, id2 = torch.min(d2_indice,1)
        # rank_weights2 = id2.float()
        
        # for k in range(d2.size(0)):
        #     rank_weights2[k]=1/(rank_weights2[k]+1)	
            
        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # cost_s = (self.margin + scores - d1)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)
        # cost_im = (self.margin + scores - d2)
        # print(cost_s[0])
        # sys.exit(0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)
 
        # keep the maximum violating negative for each query
        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]

        # weight similarity scores
        cost_s= torch.mul(rank_weights1, cost_s)
        cost_im= torch.mul(rank_weights2, cost_im)

        # print(cost_s)
        # sys.exit(0)

        return cost_s.sum() + cost_im.sum()


        
class VSE(object):
    """
    rkiros/uvs model
    """

    def __init__(self, opt):
        # tutorials/09 - Image Captioning
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.input_dim, opt.embed_size,
                                    use_abs=opt.use_abs,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   use_abs=opt.use_abs,
                                   no_imgnorm=opt.no_imgnorm)

        # pretrain embeding
        if opt.word2vec_path:
            np_weights = np.load(opt.word2vec_path)
            t_weights = torch.from_numpy(np_weights)
            self.txt_enc.embed.load_state_dict({'weight': t_weights})
            self.txt_enc.embed.weight.requires_grad = opt.word2vec_trainable

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = Loss(margin=opt.margin,
                            measure=opt.measure,
                            max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())
        # if opt.finetune:
        #     params += list(self.img_enc.cnn.parameters())
        params = filter(lambda p: p.requires_grad, params)
        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate, weight_decay=opt.weight_decay)
        # self.optimizer = torch.optim.RMSprop(params, lr=opt.learning_rate, weight_decay=1e-8)
        # self.optimizer = torch.optim.RMSprop(params, lr=opt.learning_rate, weight_decay=opt.weight_decay)
        self.Eiters = 0

    def state_dict(self):
        # remove pretrain weights
        txt_untrainable_param = [name for name, param in self.txt_enc.named_parameters() if not param.requires_grad]
        txt_state_dict = self.txt_enc.state_dict()
        for name in txt_untrainable_param:
            txt_state_dict.pop(name)

        state_dict = [self.img_enc.state_dict(), txt_state_dict]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1], strict=False)

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, captions, lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images, volatile=volatile)
        captions = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, lengths)
        return img_emb, cap_emb
        
    def forward_emb_image(self, images, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images, volatile=volatile)

        if torch.cuda.is_available():
            images = images.cuda()
            #captions = captions.cuda()

        # Forward
        img_emb = self.img_enc(images)
        return img_emb

    def forward_emb_caption(self, captions, lengths, volatile=False):
        #"""Compute the image and caption embeddings"""
        # Set mini-batch dataset
        captions = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            captions = captions.cuda()

        # Forward
        cap_emb = self.txt_enc(captions, lengths)
        return cap_emb
        
    def forward_loss(self, img_emb, cap_emb, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        #print(img_emb)
        #print(cap_emb)
        loss = self.criterion(img_emb, cap_emb)
        self.logger.update('Le', loss.data[0], img_emb.size(0))
        return loss

    def train_emb(self, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        """
        #print(ids)
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb = self.forward_emb(images, captions, lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb)

        # compute gradient and do SGD step
        # loss.register_hook(print)
        # loss.retain_grad()
        loss.backward()

        # p = list(self.img_enc.fc2.parameters())
        # print("loss:", loss.requires_grad)
        # print("grad: ", p[0].grad)
        # print(list(self.txt_enc.parameters())[0][0].data.cpu().numpy())
        # print(list(self.txt_enc.parameters())[0].grad)
        # print(list(self.img_enc.parameters())[0][0].data.cpu().numpy())
        # sys.exit(0)
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()

np.set_printoptions(threshold=sys.maxsize)