# Training Code Adapted from PyTorch Code for "VSE++: Improving Visual-Semantic Embeddings with Hard Negatives"
import pickle
import os
import time
import shutil

import torch


import data_resnet as data_obj
import data_i3d_audio as data_act
from vocabulary import Vocabulary  # NOQA
from model import VSE
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, evalrank

import logging
import tensorboard_logger as tb_logger
from constant import CONSTANT


def main():
    print(CONSTANT)

    if CONSTANT.mode == 'test' and CONSTANT.model == 'both':
        evalrank()
        return

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.unconfigure()
    tb_logger.configure(CONSTANT.log_path, flush_secs=5)
    
    # Load Vocabulary Wrapper
    with open(CONSTANT.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        # dict or Vocabulary
        if isinstance(vocab, dict):
            CONSTANT.vocab_size = len(vocab)
        else:
            CONSTANT.vocab_size = vocab.idx
        del vocab
    

    if CONSTANT.model == 'obj':
        data = data_obj
    else:
        data = data_act

    if CONSTANT.mode == 'train':
        # Load data loaders
        train_loader, val_loader = data.get_loaders()
    else:
        test_loader = data.get_test_loader()

    # Construct the model
    model = VSE(CONSTANT)

    # optionally resume from a checkpoint
    start_epoch = 0
    end_epoch = CONSTANT.num_epochs
    best_rsum = 0
    if CONSTANT.resume:
        if os.path.isfile(CONSTANT.resume):
            print("=> loading checkpoint '{}'".format(CONSTANT.resume))
            checkpoint = torch.load(CONSTANT.resume)
            start_epoch = checkpoint['epoch']
            end_epoch = start_epoch + CONSTANT.num_epochs
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(CONSTANT.resume, start_epoch, best_rsum))
        else:
            print("=> no checkpoint found at '{}'".format(CONSTANT.resume))

    if CONSTANT.mode == 'train':
        # Train the Model
        for epoch in range(start_epoch, end_epoch):
            adjust_learning_rate(model.optimizer, epoch)

            # train for one epoch
            train(train_loader, model, epoch, val_loader)

            # evaluate on validation set
            rsum = validate_tb(val_loader, model)

            # remember best R@ sum and save checkpoint
            is_best = rsum > best_rsum
            best_rsum = max(rsum, best_rsum)
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                'opt': CONSTANT,
                'Eiters': model.Eiters,
            }, is_best, prefix=CONSTANT.log_path + '/')

        ## write stat to file
        write_stat_file()
    else:
        # evaluate on validation set
        validate_tb(test_loader, model)

def write_stat_file():
    # load best model
    path = os.path.join(CONSTANT.log_path, "model_best.pth.tar")
    checkpoint = torch.load(path)
    model = VSE(checkpoint['opt'])
    model.load_state_dict(checkpoint['model'])
    epoch = checkpoint['epoch']

    # get loader
    if CONSTANT.model == 'obj':
        data = data_obj
    else:
        data = data_act
    
    train_loader, val_loader = data.get_loaders()
    test_loader = data.get_test_loader()

    # run validate
    validate_file(train_loader, model, epoch, os.path.join(CONSTANT.log_path, "train.stat"))
    validate_file(val_loader, model, epoch, os.path.join(CONSTANT.log_path, "val.stat"))
    validate_file(test_loader, model, epoch, os.path.join(CONSTANT.log_path, "test.stat"))


def train(train_loader, model, epoch, val_loader):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    # switch to train mode
    model.train_start()

    end = time.time()
    for i, train_data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        model.train_emb(*train_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % CONSTANT.log_step == 0:
            # print(model.params)
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)

        # validate at every val_step
        if model.Eiters % CONSTANT.val_step == 0:
            validate_tb(val_loader, model)

def validate(data_loader, model):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs = encode_data(
        model, data_loader, CONSTANT.log_step, logging.info)

    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, measure=CONSTANT.measure)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    # image retrieval
    (r1i, r5i, r10i, medri, meanri) = t2i(
        img_embs, cap_embs, measure=CONSTANT.measure)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanri))
    # sum of recalls to be used for early stopping
    rsum = r1 + r5 + r10 + r1i + r5i + r10i

    return rsum, r1, r5, r10, medr, meanr, r1i, r5i, r10i, medri, meanri

def validate_file(data_loader, model, epoch, file_path):
    rsum, r1, r5, r10, medr, meanr, r1i, r5i, r10i, medri, meanri = validate(data_loader, model)
    log_str =   f'epoch={epoch}\n' \
                f'r1={r1}\nr5={r5}\nr10={r10}\nmedr={medr}\nmeanr={meanr}\n' \
                f'r1i={r1i}\nr5i={r5i}\nr10i={r10i}\nmedri={medri}\nmeanri={meanri}\n' \
                f'rsum={rsum}'
        
    with open(file_path, 'w') as f:
        f.write(log_str)

def validate_tb(data_loader, model):
    rsum, r1, r5, r10, medr, meanr, r1i, r5i, r10i, medri, meanri = validate(data_loader, model)

    # record metrics in tensorboard
    tb_logger.log_value('r1', r1, step=model.Eiters)
    tb_logger.log_value('r5', r5, step=model.Eiters)
    tb_logger.log_value('r10', r10, step=model.Eiters)
    tb_logger.log_value('medr', medr, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanri', meanri, step=model.Eiters)
    tb_logger.log_value('rsum', rsum, step=model.Eiters)

    return rsum


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = CONSTANT.learning_rate * (0.5 ** (epoch // CONSTANT.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
