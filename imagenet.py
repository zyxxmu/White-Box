import torch
import torch.nn as nn
import torch.optim as optim
from utils.options import args
import utils.common as utils

import os
import time
import math
from data import imagenet
from importlib import import_module
from model.resnet import Bottleneck_class

from utils.common import *

import thop
from thop import profile

device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
checkpoint = utils.checkpoint(args)
now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
logger = utils.get_logger(os.path.join(args.job_dir, 'logger'+now+'.log'))
if args.criterion == 'Softmax':
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
else:
    raise ValueError('invalid criterion : {:}'.format(args.criterion))

# Data
print('==> Preparing data..')
data_tmp = imagenet.Data(args)
train_loader = data_tmp.trainLoader
val_loader = data_tmp.testLoader

# Architecture
if args.arch == 'resnet':
    origin_model = import_module(f'model.{args.arch}').resnet(args.cfg).to(device)
else:
    raise('arch not exist!')

# Calculate FLOPs of origin model
input = torch.randn(1, 3, 224, 224).to(device)
oriflops, oriparams = profile(origin_model, inputs=(input, ))

# Based on the trained class-wise mask, perform global voting to obtain pruned model
def build_resnet_pruned_model(origin_model):

    pruning_rate_now = 0
    channel_prune_rate = 0.9
    num_mask_cfg = {'resnet50' : 48}

    while pruning_rate_now < args.pruning_rate:

        score = []
        index_cfg = []
        block_index_cfg = []
        layer_cfg = []
        block_cfg = []
        final_mask = []
        pruned_state_dict = {}

        for i in range(num_mask_cfg[args.cfg]):
            mask = origin_model.state_dict()['mask.'+str(i)]
            score.append(torch.abs(torch.sum(mask, 0)))
            final_mask.append(torch.div(torch.sum(mask, 0), 2))

        all_score = torch.cat(score,0)

        preserve_num = int(all_score.size(0) * channel_prune_rate)
        preserve_channel, _ = torch.topk(all_score, preserve_num)

        threshold = preserve_channel[preserve_num-1]
        
        block_score = []

        # Based on the pruning threshold, the prune cfg of each layer is obtained
        for i, mini_score in enumerate(score):
            mask = torch.ge(mini_score, threshold)
            index = []
            for j, m in enumerate(mask):
                if m == True:
                    index.append(j)
            if len(index) < mask.size(0) * args.min_preserve:
                _, index = torch.topk(mini_score, int(mask.size(0) * args.min_preserve))
                index = index.cpu().numpy().tolist()
            if (i + 1) % 3 != 0: #in block
                index_cfg.append(index)
                layer_cfg.append(len(index))
            else: #out block
                block_score.append(mini_score)

        num_blocks = [3,4,6,3]
        begin = 0
        for i in range(len(num_blocks)):
            block_cfg.append(int(block_score[begin].size(0)/4))
            for j in range(begin, begin + num_blocks[i]):
                block_index_cfg.append(torch.arange(block_score[begin].size(0)))
            begin = begin + num_blocks[i]
        
        model = import_module(f'model.{args.arch}').resnet(args.cfg, block_cfg, layer_cfg).to(device)

        flops, params = profile(model, inputs=(input, ))

        pruning_rate_now = (oriflops - flops) / oriflops

        channel_prune_rate = channel_prune_rate - 0.01

    model_state_dict = origin_model.state_dict()

    current_block = 0

    block_index = torch.arange(64)

    model = import_module(f'model.{args.arch}').resnet(args.cfg, block_cfg, layer_cfg).to(device)
    
    pruned_state_dict = model.state_dict()
    
    for name, module in origin_model.named_modules():

        if isinstance(module, Bottleneck_class):
        
            # conv1 & bn1
            index_1 = torch.LongTensor(index_cfg[current_block * 2]).to(device)
            
            pruned_weight = torch.index_select(model_state_dict[name + '.conv1.weight'], 0, index_1).cpu()
            pruned_weight = direct_project(pruned_weight, block_index)

            pruned_state_dict[name + '.conv1.weight'] = pruned_weight

            mask = final_mask[current_block * 3][index_cfg[current_block * 2]]
            pruned_state_dict[name + '.bn1.weight'] = torch.mul(mask,model_state_dict[name + '.bn1.weight'][index_1]).cpu()
            pruned_state_dict[name + '.bn1.bias'] = torch.mul(mask,model_state_dict[name + '.bn1.bias'][index_1]).cpu()
            pruned_state_dict[name + '.bn1.running_var'] = model_state_dict[name + '.bn1.running_var'][index_1].cpu()
            pruned_state_dict[name + '.bn1.running_mean'] = model_state_dict[name + '.bn1.running_mean'][index_1].cpu()

            # conv2 & bn2
            index_2 = torch.LongTensor(index_cfg[current_block * 2 + 1]).to(device)
            
            pruned_weight = torch.index_select(model_state_dict[name + '.conv2.weight'], 0, index_2).cpu()
            pruned_weight = direct_project(pruned_weight, index_1)

            pruned_state_dict[name + '.conv2.weight'] = pruned_weight

            mask = final_mask[current_block * 3 + 1][index_cfg[current_block * 2 + 1]]
            pruned_state_dict[name + '.bn2.weight'] = torch.mul(mask,model_state_dict[name + '.bn2.weight'][index_2]).cpu()
            pruned_state_dict[name + '.bn2.bias'] = torch.mul(mask,model_state_dict[name + '.bn2.bias'][index_2]).cpu()
            pruned_state_dict[name + '.bn2.running_var'] = model_state_dict[name + '.bn2.running_var'][index_2].cpu()
            pruned_state_dict[name + '.bn2.running_mean'] = model_state_dict[name + '.bn2.running_mean'][index_2].cpu()

            block_index = torch.LongTensor(block_index_cfg[current_block]).to(device)
            mask = final_mask[current_block * 3 + 2][block_index_cfg[current_block]]

            # conv3 & bn3 & shortcut
            
            pruned_state_dict[name + '.conv3.weight'] = torch.index_select(model_state_dict[name + '.conv3.weight'], 0, block_index).cpu()
            pruned_state_dict[name + '.conv3.weight'] = direct_project(pruned_state_dict[name + '.conv3.weight'], index_2)
            pruned_state_dict[name + '.bn3.weight'] = model_state_dict[name + '.bn3.weight'].cpu()
            pruned_state_dict[name + '.bn3.bias'] = model_state_dict[name + '.bn3.bias'].cpu()
            pruned_state_dict[name + '.bn3.running_var'] = model_state_dict[name + '.bn3.running_var'][block_index].cpu()
            pruned_state_dict[name + '.bn3.running_mean'] = model_state_dict[name + '.bn3.running_mean'][block_index].cpu()

            current_block += 1
    
    pruned_state_dict['fc.weight'] = model_state_dict['fc.weight'].cpu()
    pruned_state_dict['fc.bias'] = model_state_dict['fc.bias'].cpu()

    pruned_state_dict['conv1.weight'] = model_state_dict['conv1.weight'].cpu()
    pruned_state_dict['bn1.weight'] = model_state_dict['bn1.weight'].cpu()
    pruned_state_dict['bn1.bias'] = model_state_dict['bn1.bias'].cpu()
    pruned_state_dict['bn1.running_var'] = model_state_dict['bn1.running_var'].cpu()
    pruned_state_dict['bn1.running_mean'] = model_state_dict['bn1.running_mean'].cpu()

    #load weight
    model = import_module(f'model.{args.arch}').resnet(args.cfg, block_cfg = block_cfg, layer_cfg=layer_cfg).to(device)
    model.load_state_dict(pruned_state_dict)
    return model, [layer_cfg, block_cfg], flops, params

# Train class-wise mask 
def train_class(epoch, train_loader, model, criterion, optimizer):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    model.train()
    end = time.time()

    num_iter = len(train_loader)

    print_freq = num_iter // 10
    i = 0 
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        if args.debug:
            if i > 5:
                break
            i += 1
        images = images.to(device)
        targets = targets.to(device)
        data_time.update(time.time() - end)

        # compute output
        logits, mask = model(images, targets)
        loss = criterion(logits, targets)
        for m in mask:
            loss += float(args.sparse_lambda) * torch.sum(m, 0).norm(2)

        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)  # accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % print_freq == 0 and batch_idx != 0:
            logger.info(
                'Epoch[{0}]({1}/{2}): '
                'Loss {loss.avg:.4f} '
                'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                    epoch, batch_idx, num_iter, loss=losses,
                    top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg

# Train function
def train(epoch, train_loader, model, criterion, optimizer):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    model.train()
    end = time.time()

    num_iter = len(train_loader)

    print_freq = num_iter // 10
    i = 0 
    for batch_idx, (images, targets) in enumerate(train_loader):
        if args.debug:
            if i > 5:
                break
            i += 1
        images = images.cuda()
        targets = targets.cuda()
        data_time.update(time.time() - end)

        adjust_learning_rate(optimizer, epoch, batch_idx, num_iter)

        # compute output
        logits = model(images)
        loss = criterion(logits, targets)

        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)  # accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % print_freq == 0 and batch_idx != 0:
            logger.info(
                'Epoch[{0}]({1}/{2}): '
                'Loss {loss.avg:.4f} '
                'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                    epoch, batch_idx, num_iter, loss=losses,
                    top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg

# Validate function
def validate(val_loader, model, criterion, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    num_iter = len(val_loader)

    model.eval()
    with torch.no_grad():
        end = time.time()
        i = 0
        for batch_idx, (images, targets) in enumerate(val_loader):
            if args.debug:
                if i > 5:
                    break
                i += 1
            images = images.cuda()
            targets = targets.cuda()

            # compute output
            logits = model(images)
            loss = criterion(logits, targets)

            # measure accuracy and record loss
            pred1, pred5 = utils.accuracy(logits, targets, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                    .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg

def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    #Warmup
    if args.lr_type == 'step':
        factor = epoch // 30
        if epoch >= 80:
            factor = factor + 1
        lr = args.lr * (0.1 ** factor)
    elif args.lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * (epoch - 5) / (args.num_epochs - 5)))
    elif args.lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = args.lr * (decay ** (epoch // step))
    elif args.lr_type == 'fixed':
        lr = args.lr
    else:
        raise NotImplementedError
    if step == 0:
        print('current learning rate:{0}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    start_epoch = 0
    best_top1_acc = 0.0
    best_top5_acc = 0.0
    
    print('==> Building Model..')
    if args.resume == False:
        if args.arch == 'resnet':
            model = import_module(f'model.{args.arch}').resnet_class(args.cfg).to(device)
        else:
            raise('arch not exist!')
        print("Model Construction Down!")

        # Train class-wise mask
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        for epoch in range(args.classtrain_epochs):
            print('Epoch:{}'.format(epoch))
            train_class(epoch, train_loader, model, criterion, optimizer)
            model_state_dict = model.state_dict()
            state = {
                'state_dict': model_state_dict,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'arch': args.cfg,
            }
            checkpoint.save_class_model(state)
            
        # Global voting to obtain pruned model
        print('==> Building Pruned Model..')
        if args.arch == 'resnet':
            model, layer_cfg, flops, params = build_resnet_pruned_model(model)
        else:
            raise('arch not exist!')
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    else:
        resumeckpt = torch.load(args.job_dir+'/checkpoint/model_last.pt')
        state_dict = resumeckpt['state_dict']
        cfg = resumeckpt['cfg']

        if args.arch == 'resnet':
            model = import_module(f'model.{args.arch}').resnet(args.cfg, block_cfg = cfg[1],layer_cfg=cfg[0]).to(device)
        else:
            raise('arch not exist!')

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(resumeckpt['optimizer'])
        start_epoch = resumeckpt['epoch']

    if len(args.gpus) != 1:
        model = nn.DataParallel(model, device_ids=args.gpus)

    # fine-tuning
    for epoch in range(start_epoch, args.num_epochs):
        train_obj, train_top1_acc,  train_top5_acc = train(epoch, train_loader, model, criterion, optimizer)
        valid_obj, test_top1_acc, test_top5_acc = validate(val_loader, model, criterion, args)

        is_best = best_top5_acc < test_top5_acc
        best_top1_acc = max(best_top1_acc, test_top1_acc)
        best_top5_acc = max(best_top5_acc, test_top5_acc)

        model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()

        state = {
            'state_dict': model_state_dict,
            'best_top1_acc': best_top1_acc,
            'best_top5_acc': best_top5_acc,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'arch': args.cfg,
            'cfg': layer_cfg
        }
        checkpoint.save_model(state, epoch + 1, is_best)

    logger.info('Best Top-1 accuracy: {:.3f} Top-5 accuracy: {:.3f}'.format(float(best_top1_acc), float(best_top5_acc)))

    logger.info('--------------Layer Configuration--------------')
    logger.info(layer_cfg)

    logger.info('--------------UnPrune Model--------------')
    logger.info('Params: %.2f M '%(oriparams/1000000))
    logger.info('FLOPS: %.2f M '%(oriflops/1000000))

    logger.info('--------------Prune Model--------------')
    logger.info('Params: %.2f M'%(params/1000000))
    logger.info('FLOPS: %.2f M'%(flops/1000000))

    logger.info('--------------Compress Rate--------------')
    logger.info('Params Compress Rate: %.2f M/%.2f M(%.2f%%)' % (params/1000000, oriparams/1000000, 100. * (oriparams- params) / oriparams))
    logger.info('FLOPS Compress Rate: %.2f M/%.2f M(%.2f%%)' % (flops/1000000, oriflops/1000000, 100. * (oriflops- flops) / oriflops))
    
if __name__ == '__main__':
    main()
