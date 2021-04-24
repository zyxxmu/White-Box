from __future__ import absolute_import
import datetime
import shutil
from pathlib import Path
import os 
import math
import torch
import logging
import torch.nn as nn

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
     
def ensure_path(directory):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
 
def mkdir(path):
    if not os.path.isdir(path):
        mkdir(os.path.split(path)[0])  
    else:
        return
    os.mkdir(path)

'''record configurations'''
class record_config():
    def __init__(self, args):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        today = datetime.date.today()

        self.args = args
        self.job_dir = Path(args.job_dir)

        def _make_dir(path):
            if not os.path.exists(path):
                os.makedirs(path)

        _make_dir(self.job_dir)

        config_dir = self.job_dir / 'config.txt'
        if args.resume != None:
            with open(config_dir, 'a') as f:
                f.write(now + '\n\n')
                for arg in vars(args):
                    f.write('{}: {}\n'.format(arg, getattr(args, arg)))
                f.write('\n')
        else:
            with open(config_dir, 'w') as f:
                f.write(now + '\n\n')
                for arg in vars(args):
                    f.write('{}: {}\n'.format(arg, getattr(args, arg)))
                f.write('\n')

class checkpoint():
    def __init__(self, args):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        self.args = args
        self.job_dir = Path(args.job_dir)
        self.ckpt_dir = self.job_dir / 'checkpoint'
        self.run_dir = self.job_dir / 'run'

        def _make_dir(path):
            if not os.path.exists(path):
                os.makedirs(path)

        _make_dir(self.job_dir)
        _make_dir(self.ckpt_dir)
        _make_dir(self.run_dir)

        record_config(args)

    def save_model(self, state, epoch, is_best):
        save_path = f'{self.ckpt_dir}/model_last.pt'
        torch.save(state, save_path)
        if is_best:
            shutil.copyfile(save_path, f'{self.ckpt_dir}/model_best.pt')

    def save_class_model(self, state):
        save_path = f'{self.ckpt_dir}/class_model.pt'
        # print('=> Saving model to {}'.format(save_path))
        torch.save(state, save_path)

    def save_pretrain_model(self, state):
        save_path = f'{self.ckpt_dir}/pretrain_model.pt'
        # print('=> Saving model to {}'.fo_rmat(save_path))
        torch.save(state, save_path)

def get_logger(file_path):

    logger = logging.getLogger('gal')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger

"""Computes the precision@k for the specified values of k"""
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0,keepdim=True)
            res.append(correct_k.mul_(100.0/batch_size))
        return res

def direct_project(weight, indices):
    #print(weight.size())

    A = torch.randn(weight.size(0), len(indices), weight.size(2), weight.size(3))
    #print(A.size())
    for i, indice in enumerate(indices):

        A[:, i, :, :] = weight[:, indice, :, :]

    return A
