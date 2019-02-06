from __future__ import print_function
import os
import pickle
import sys

import numpy
import time
import numpy as np

# from vocab import Vocabulary  # NOQA
import torch
from data_resnet import VTTDataset as ResnetData, collate_fn as resnet_collate
from data_i3d_audio import VTTDataset as I3DSoundData, collate_fn as i3dsound_collate
from model import VSE
from collections import OrderedDict
from constant import CONSTANT


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (0.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return "%.4f (%.4f)" % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ""
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += "  "
            s += k + " " + str(v)
        return s

    def tb_log(self, tb_logger, prefix="", step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all videos and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    for i, (videos, captions, lengths, ids) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        img_emb, cap_emb = model.forward_emb(videos, captions, lengths, volatile=True)

        # initialize the numpy arrays given the size of the embeddings
        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids] = cap_emb.data.cpu().numpy().copy()

        # measure accuracy and record loss
        model.forward_loss(img_emb, cap_emb)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging(
                "Test: [{0}/{1}]\t"
                "{e_log}\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t".format(
                    i, len(data_loader), batch_time=batch_time, e_log=str(model.logger)
                )
            )
        del videos, captions

    return img_embs, cap_embs


def load_model(model_path, cap_path, resnet_path, i3drgb_path, i3dflow_path, soundnet_path, batch_size, num_workers):
    # construct model resnet
    checkpoint = torch.load(model_path)
    opt = checkpoint["opt"]

    model = VSE(opt)
    start_epoch = checkpoint['epoch']
    best_rsum = checkpoint['best_rsum']
    model.load_state_dict(checkpoint['model'])
    model.Eiters = checkpoint['Eiters']
    print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
            .format(model_path, start_epoch, best_rsum))

    if opt.model == 'obj':
        if opt.input_dim == 1024:
            dataset = ResnetData(np_cap=cap_path, resnet_path=i3dflow_path)
        else:
            dataset = ResnetData(np_cap=cap_path, resnet_path=resnet_path)
        collate_fn = resnet_collate
    else:
        collate_fn = i3dsound_collate
        if opt.input_dim == 3072:
            dataset = I3DSoundData(np_cap=cap_path,
                rgbi3d_path=i3drgb_path, soundnet_path=soundnet_path, 
                flowi3d_path=i3dflow_path)
        else:
            dataset = I3DSoundData(np_cap=cap_path,
                rgbi3d_path=i3drgb_path, soundnet_path=soundnet_path, 
                flowi3d_path=None)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False
    )

    return model, data_loader

def evalrank():
    """
    Evaluate a trained model.
    """
    cpv = CONSTANT.cpv  # caps per video
    shared_space = 'two'
    # construct model resnet
    print("Loading model and dataset 1")
    model1, data_loader1 = load_model(CONSTANT.model_path1, CONSTANT.cap_test_path,
        CONSTANT.resnet_path, CONSTANT.rgbi3d_path, CONSTANT.flowi3d_path, CONSTANT.soundnet_path,
        CONSTANT.batch_size, CONSTANT.workers)

    print("Computing results 1")
    img_embs1, cap_embs1 = encode_data(model1, data_loader1)

    # construct model i3d-audio
    print("Loading model and dataset 2")
    model2, data_loader2 = load_model(CONSTANT.model_path2, CONSTANT.cap_test_path,
        CONSTANT.resnet_path, CONSTANT.rgbi3d_path, CONSTANT.flowi3d_path, CONSTANT.soundnet_path,
        CONSTANT.batch_size, CONSTANT.workers)

    print("Computing results 2")
    img_embs2, cap_embs2 = encode_data(model2, data_loader2)

    # third model
    img_embs3 = None
    cap_embs3 = None
    if CONSTANT.model_path3:
        print("Loading model and dataset 3")
        model3, data_loader3 = load_model(CONSTANT.model_path3, CONSTANT.cap_test_path,
            CONSTANT.resnet_path, CONSTANT.rgbi3d_path, CONSTANT.flowi3d_path, CONSTANT.soundnet_path,
            CONSTANT.batch_size, CONSTANT.workers)

        print("Computing results 3")
        img_embs3, cap_embs3 = encode_data(model3, data_loader3)
        shared_space = 'three'

    # print total images and captions
    print("Images: %d, Captions: %d" % (img_embs2.shape[0] // cpv, cap_embs2.shape[0]))

    # no cross-validation, full evaluation
    r, rt = i2t(
        img_embs1,
        cap_embs1,
        img_embs2,
        cap_embs2,
        img_embs3,
        cap_embs3,
        shared_space=shared_space,
        measure=CONSTANT.measure,
        return_ranks=True,
    )

    ri, rti = t2i(
        img_embs1,
        cap_embs1,
        img_embs2,
        cap_embs2,
        img_embs3,
        cap_embs3,
        shared_space=shared_space,
        measure=CONSTANT.measure,
        return_ranks=True,
    )

    ar = (r[0] + r[1] + r[2]) / 3
    ari = (ri[0] + ri[1] + ri[2]) / 3
    rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
    print("rsum: %.2f" % rsum)
    print("Average i2t Recall: %.2f" % ar)
    print("Image to text: %.2f %.2f %.2f %.2f %.2f" % r)
    print("Average t2i Recall: %.2f" % ari)
    print("Text to image: %.2f %.2f %.2f %.2f %.2f" % ri)

def i2t(
    videos,
    captions,
    videos2=None,
    captions2=None,
    videos3=None,
    captions3=None,
    shared_space="one_space",
    measure="cosine",
    return_ranks=False,
):
    """
    Videos->Text (Video Annotation)
    Videos: (20N, K) matrix of videos
    Captions: (20N, K) matrix of captions
    """

    cpv = CONSTANT.cpv  # caps per video
    npts = videos.shape[0] // cpv
    index_list = []

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    for index in range(npts):
        # Get query image
        im = videos[cpv * index].reshape(1, videos.shape[1])

        # Compute scores
        if "three" == shared_space:
            im2 = videos2[cpv * index].reshape(1, videos2.shape[1])
            im3 = videos3[cpv * index].reshape(1, videos3.shape[1])
            d1 = numpy.dot(im, captions.T).flatten()
            d2 = numpy.dot(im2, captions2.T).flatten()
            d3 = numpy.dot(im3, captions3.T).flatten()
            d = d1 + d2 + d3
        elif "two" == shared_space:
            im2 = videos2[cpv * index].reshape(1, videos2.shape[1])
            d1 = numpy.dot(im, captions.T).flatten()
            d2 = numpy.dot(im2, captions2.T).flatten()
            d = d1 + d2
        else:
            d = numpy.dot(im, captions.T).flatten()
        # elif 'object_text' == shared_space:
        #     d = numpy.dot(im, captions.T).flatten()
        # elif 'activity_text' == shared_space:
        # d = numpy.dot(im2, captions2.T).flatten()

        inds = numpy.argsort(d)[::-1]
        index_list.append(inds[0])
        # Score
        rank = 1e20
        for i in range(cpv * index, cpv * index + cpv, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
                flag = i - cpv * index
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(
    videos,
    captions,
    videos2=None,
    captions2=None,
    videos3=None,
    captions3=None,
    shared_space="one_space",
    measure="cosine",
    return_ranks=False,
):
    """
    Text->Videos (Video Search)
    Videos: (20N, K) matrix of videos
    Captions: (20N, K) matrix of captions
    """

    cpv = CONSTANT.cpv  # caps per video
    npts = videos.shape[0] // cpv
    ims = numpy.array([videos[i] for i in range(0, len(videos), cpv)])
    # ims2 = numpy.array([videos2[i] for i in range(0, len(videos2), 20)])
    if "three" == shared_space:
        ims3 = numpy.array([videos3[i] for i in range(0, len(videos3), cpv)])
        ims2 = numpy.array([videos2[i] for i in range(0, len(videos2), cpv)])
    elif "two" == shared_space:
        ims2 = numpy.array([videos2[i] for i in range(0, len(videos2), cpv)])

    ranks = numpy.zeros(cpv * npts)
    top1 = numpy.zeros(cpv * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[cpv * index : cpv * index + cpv]
        # queries2 = captions2[20 * index:20 * index + 20]

        if "three" == shared_space:
            queries2 = captions2[cpv * index : cpv * index + cpv]
            queries3 = captions3[cpv * index : cpv * index + cpv]
            d1 = numpy.dot(queries, ims.T)
            d2 = numpy.dot(queries2, ims2.T)
            d3 = numpy.dot(queries3, ims3.T)
            d = d1 + d2 + d3
        elif "two" == shared_space:
            queries2 = captions2[cpv * index : cpv * index + cpv]
            d1 = numpy.dot(queries, ims.T)
            d2 = numpy.dot(queries2, ims2.T)
            d = d1 + d2
        else:
            d = numpy.dot(queries, ims.T)
        # elif 'object_text' == shared_space:
        #     d = numpy.dot(queries, ims.T)
        # elif 'activity_text' == shared_space:
        #     d = numpy.dot(queries2, ims2.T)

        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            ranks[cpv * index + i] = numpy.where(inds[i] == index)[0][0]
            top1[cpv * index + i] = inds[i][0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)
