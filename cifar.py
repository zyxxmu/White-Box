import torch
import torch.nn as nn
import torch.optim as optim
from utils.options import args
import utils.common as utils

import os
import time
import math
import random
import numpy
from data import cifar10
from importlib import import_module
from model.resnet_cifar import ResBasicBlock_Class
from model.mobilenet_v2 import Block_class

from thop import profile

from utils.common import *
import sys

device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
checkpoint = utils.checkpoint(args)
logger = utils.get_logger(os.path.join(args.job_dir + 'logger.log'))
loss_func = nn.CrossEntropyLoss()

# Data
print('==> Preparing data..')
loader = cifar10.Data(args)

# Architecture
if args.arch == 'vgg_cifar':
    origin_model = import_module(f'model.{args.arch}').VGG(args.cfg).to(device)
elif args.arch == 'resnet_cifar':
    origin_model = import_module(f'model.{args.arch}').resnet(args.cfg).to(device)
elif args.arch == 'mobilenet_v2':
    origin_model = import_module(f'model.{args.arch}').MobileNetV2(wm=1).to(device)

# Calculate FLOPs of origin model
input = torch.randn(1, 3, 32, 32).to(device)
oriflops, oriparams = profile(origin_model, inputs=(input, ))

# Based on the trained class-wise mask, perform global voting to obtain pruned model
def build_mobilenetv2_pruned_model(origin_model):
    pruning_rate_now = 0
    channel_prune_rate = 0.9

    while pruning_rate_now < args.pruning_rate:
        score = []
        index_cfg = []
        layer_cfg = []
        final_mask = []
        pruned_state_dict = {}

        # Get importance criteria for each channel
        for i in range(17):
            mask = origin_model.state_dict()['mask.'+str(i)]
            score.append(torch.abs(torch.div(torch.sum(mask, 0), 2)))
            final_mask.append(torch.div(torch.sum(mask, 0), 2))

        all_score = torch.cat(score,0)

        preserve_num = int(all_score.size(0) * channel_prune_rate)
        preserve_channel, _ = torch.topk(all_score, preserve_num)

        threshold = preserve_channel[preserve_num-1]

        # Based on the pruning threshold, the pruning rate of each layer is obtained
        for mini_score in score:
            mask = torch.ge(mini_score, threshold)
            index = []
            for i, m in enumerate(mask):
                if m == True:
                    index.append(i)
            if len(index) < mask.size(0) * args.min_preserve:
                _, index = torch.topk(mini_score, int(mask.size(0) * args.min_preserve))
                index = index.cpu().numpy().tolist()
            index_cfg.append(index)
            layer_cfg.append(len(index))
        layer_cfg.append(640)
        last_index = random.sample(range(0, 1280), 640)
        last_index.sort()
        index_cfg.append(last_index)
        last_index = torch.LongTensor(last_index).to(device)


        model = import_module(f'model.{args.arch}').MobileNetV2(wm=1,layer_cfg=layer_cfg).to(device)
        flops, params = profile(model, inputs=(input, ))
        pruning_rate_now = (oriflops - flops) / oriflops
        channel_prune_rate = channel_prune_rate - 0.01

    model_state_dict = origin_model.state_dict()
    current_layer = 0
    
    model = import_module(f'model.{args.arch}').MobileNetV2(wm=1,layer_cfg=layer_cfg).to(device)
    pruned_state_dict = model.state_dict()
    for name, module in origin_model.named_modules():
        if isinstance(module, Block_class):
            # conv1 & bn1
            index = torch.LongTensor(index_cfg[current_layer]).to(device)
            pruned_weight = torch.index_select(model_state_dict[name + '.conv1.weight'], 0, index).cpu()
            pruned_state_dict[name + '.conv1.weight'] = pruned_weight
            mask = final_mask[current_layer][index_cfg[current_layer]]
            pruned_state_dict[name + '.bn1.weight'] = torch.mul(mask,model_state_dict[name + '.bn1.weight'][index]).cpu()
            pruned_state_dict[name + '.bn1.bias'] = torch.mul(mask,model_state_dict[name + '.bn1.bias'][index]).cpu()
            pruned_state_dict[name + '.bn1.running_var'] = model_state_dict[name + '.bn1.running_var'][index].cpu()
            pruned_state_dict[name + '.bn1.running_mean'] = model_state_dict[name + '.bn1.running_mean'][index].cpu()

            # conv2 & bn2
            pruned_weight = torch.index_select(model_state_dict[name + '.conv2.weight'], 0, index).cpu()
            pruned_state_dict[name + '.conv2.weight'] = pruned_weight
            pruned_state_dict[name + '.bn2.weight'] = torch.mul(mask,model_state_dict[name + '.bn2.weight'][index]).cpu()
            pruned_state_dict[name + '.bn2.bias'] = torch.mul(mask,model_state_dict[name + '.bn2.bias'][index]).cpu()
            pruned_state_dict[name + '.bn2.running_var'] = model_state_dict[name + '.bn2.running_var'][index].cpu()
            pruned_state_dict[name + '.bn2.running_mean'] = model_state_dict[name + '.bn2.running_mean'][index].cpu()

            # conv3 & bn3 & shortcut
            pruned_state_dict[name + '.conv3.weight'] = direct_project(model_state_dict[name + '.conv3.weight'], index).cpu()
            pruned_state_dict[name + '.bn3.weight'] = model_state_dict[name + '.bn3.weight'].cpu()
            pruned_state_dict[name + '.bn3.bias'] = model_state_dict[name + '.bn3.bias'].cpu()
            pruned_state_dict[name + '.bn3.running_var'] = model_state_dict[name + '.bn3.running_var'].cpu()
            pruned_state_dict[name + '.bn3.running_mean'] = model_state_dict[name + '.bn3.running_mean'].cpu()

            current_layer += 1
    

    pruned_state_dict['conv1.weight'] = model_state_dict['conv1.weight'].cpu()
    pruned_state_dict['bn1.weight'] = model_state_dict['bn1.weight'].cpu()
    pruned_state_dict['bn1.bias'] = model_state_dict['bn1.bias'].cpu()
    pruned_state_dict['bn1.running_var'] = model_state_dict['bn1.running_var'].cpu()
    pruned_state_dict['bn1.running_mean'] = model_state_dict['bn1.running_mean'].cpu()

    pruned_state_dict['conv2.weight'] = torch.index_select(model_state_dict['conv2.weight'],0,last_index).cpu()
    pruned_state_dict['bn2.weight'] = model_state_dict['bn2.weight'][last_index].cpu()
    pruned_state_dict['bn2.bias'] = model_state_dict['bn2.bias'][last_index].cpu()
    pruned_state_dict['bn2.running_var'] = model_state_dict['bn2.running_var'][last_index].cpu()
    pruned_state_dict['bn2.running_mean'] = model_state_dict['bn2.running_mean'][last_index].cpu()
    
    fc_weight = model_state_dict['linear.weight']
    pr_fc_weight = torch.randn(fc_weight.size(0),len(last_index))
    for i, ind in enumerate(last_index):
        pr_fc_weight[:,i] = fc_weight[:,ind]
    pruned_state_dict['linear.weight'] = pr_fc_weight.cpu()
    pruned_state_dict['linear.bias'] = model_state_dict['linear.bias']
    
    # load weight
    model = import_module(f'model.{args.arch}').MobileNetV2(wm=1,layer_cfg=layer_cfg).to(device)
    model.load_state_dict(pruned_state_dict)

    return model, layer_cfg, flops, params

# Based on the trained class-wise mask, perform global voting to obtain pruned model
def build_vgg_pruned_model(origin_model):
    pruning_rate_now = 0
    channel_prune_rate = 0.9

    while pruning_rate_now < args.pruning_rate:
        score = []
        index_cfg = []
        layer_cfg = []
        final_mask = []
        pruned_state_dict = {}

        # Get importance criteria for each channel
        for i in range(12):
            mask = origin_model.state_dict()['mask.'+str(i)]
            score.append(torch.abs(torch.div(torch.sum(mask, 0), 2)))
            final_mask.append(torch.div(torch.sum(mask, 0), 2))
        all_score = torch.cat(score,0)
        preserve_num = int(all_score.size(0) * channel_prune_rate)
        preserve_channel, _ = torch.topk(all_score, preserve_num)
        threshold = preserve_channel[preserve_num-1]

        # Based on the pruning threshold, the pruning rate of each layer is obtained
        for mini_score in score:
            mask = torch.ge(mini_score, threshold)
            index = []
            for i, m in enumerate(mask):
                if m == True:
                    index.append(i)
            if len(index) < mask.size(0) * args.min_preserve:
                _, index = torch.topk(mini_score, int(mask.size(0) * args.min_preserve))
                index = index.cpu().numpy().tolist()
            index_cfg.append(index)
            layer_cfg.append(len(index))

        last_layer_cfg = int(512 * (1 - pruning_rate_now))
        last_index = random.sample(range(0, 512), last_layer_cfg)
        last_index.sort()
        index_cfg.append(last_index)
        layer_cfg.append(last_layer_cfg)

        model = import_module(f'model.{args.arch}').VGG(args.cfg, layer_cfg=layer_cfg).to(device)

        # Update current pruning rate \alpha
        flops, params = profile(model, inputs=(input, ))
        pruning_rate_now = (oriflops - flops) / oriflops
        channel_prune_rate = channel_prune_rate - 0.01

    model_state_dict = origin_model.state_dict()
    current_layer = 0

    for name, module in origin_model.named_modules():
        if isinstance(module, nn.Conv2d):
            index = torch.LongTensor(index_cfg[current_layer]).to(device)
            pruned_weight = torch.index_select(model_state_dict[name + '.weight'], 0, index).cpu()
            pruned_bias = model_state_dict[name + '.bias'][index].cpu()
            pruned_state_dict[name + '.weight'] = pruned_weight
            pruned_state_dict[name + '.bias'] = pruned_bias

        elif isinstance(module, nn.BatchNorm2d):
            if current_layer == 12:
                pruned_state_dict[name + '.weight'] = model_state_dict[name + '.weight'][index].cpu()
                pruned_state_dict[name + '.bias'] = model_state_dict[name + '.bias'][index].cpu()
            else:
                mask = final_mask[current_layer][index_cfg[current_layer]]
                pruned_state_dict[name + '.weight'] = torch.mul(mask,model_state_dict[name + '.weight'][index]).cpu()
                pruned_state_dict[name + '.bias'] = torch.mul(mask,model_state_dict[name + '.bias'][index]).cpu()
            pruned_state_dict[name + '.running_var'] = model_state_dict[name + '.running_var'][index].cpu()
            pruned_state_dict[name + '.running_mean'] = model_state_dict[name + '.running_mean'][index].cpu()
            current_layer += 1

    # load weight
    model = import_module(f'model.{args.arch}').VGG(args.cfg, layer_cfg=layer_cfg).to(device)
    current_layer = 0
    for i, (k, v) in enumerate(pruned_state_dict.items()):
        weight = torch.FloatTensor(pruned_state_dict[k])
        if i == 0: # first conv need not to prune channel
            continue
        if weight.dim() == 4: # conv_layer
            pruned_state_dict[k] = direct_project(weight, index_cfg[current_layer])
            current_layer += 1

    fc_weight = model_state_dict['classifier.weight']
    pr_fc_weight = torch.randn(fc_weight.size(0),len(index))
    for i, ind in enumerate(index):
        pr_fc_weight[:,i] = fc_weight[:,ind]
    pruned_state_dict['classifier.weight'] = pr_fc_weight.cpu()
    pruned_state_dict['classifier.bias'] = model_state_dict['classifier.bias']

    model = import_module(f'model.{args.arch}').VGG(args.cfg, layer_cfg=layer_cfg).to(device)
    model.load_state_dict(pruned_state_dict)

    return model, layer_cfg, flops, params

# Based on the trained class-wise mask, perform global voting to obtain pruned model         
def build_resnet_pruned_model(origin_model):
    pruning_rate_now = 0
    channel_prune_rate = 0.9
    num_mask_cfg = {'resnet56' : 55, 'resnet110' : 109}
    while pruning_rate_now < args.pruning_rate:
        score = []
        index_cfg = []
        block_index_cfg = []
        layer_cfg = []
        block_cfg = []
        final_mask = []
        pruned_state_dict = {}

        # Get importance criteria for each channel
        for i in range(num_mask_cfg[args.cfg]):
            mask = origin_model.state_dict()['mask.'+str(i)]
            score.append(torch.abs(torch.sum(mask, 0)))
            final_mask.append(torch.div(torch.sum(mask,0), 2))

        all_score = torch.cat(score,0)
        preserve_num = int(all_score.size(0) * channel_prune_rate)
        preserve_channel, _ = torch.topk(all_score, preserve_num)

        threshold = preserve_channel[preserve_num-1]
        
        block_score = []
        #Based on the pruning threshold, the pruning rate of each layer is obtained
        for i, mini_score in enumerate(score):
            mask = torch.ge(mini_score, threshold)
            index = []
            for j, m in enumerate(mask):
                if m == True:
                    index.append(j)
            if len(index) < mask.size(0) * args.min_preserve:
                _, index = torch.topk(mini_score, int(mask.size(0) * args.min_preserve))
                index = index.cpu().numpy().tolist()
            if i % 2 != 0: # in block
                index_cfg.append(index)
                layer_cfg.append(len(index))
            else: # out block
                block_score.append(mini_score)

        begin = 0
        end = int(num_mask_cfg[args.cfg]/6) + 1
        
        for i in range(3):
            block_cfg.append(block_score[begin].size(0))
            for i in block_score[begin : end]:
                block_index_cfg.append(torch.arange(block_score[begin].size(0)))
            begin = end
            end = end + int(num_mask_cfg[args.cfg]/6)

        model = import_module(f'model.{args.arch}').resnet(args.cfg, layer_cfg, block_cfg).to(device)

        flops, params = profile(model, inputs=(input, ))
        pruning_rate_now = (oriflops - flops) / oriflops
        channel_prune_rate = channel_prune_rate - 0.01

    model_state_dict = origin_model.state_dict()
    current_block = 0

    block_index = torch.LongTensor(block_index_cfg[0]).to(device)
    mask = final_mask[0][block_index_cfg[0]]
    pruned_state_dict['conv1.weight'] = torch.index_select(model_state_dict['conv1.weight'], 0, block_index).cpu()
    pruned_state_dict['bn1.weight'] = torch.mul(mask, model_state_dict['bn1.weight'][block_index]).cpu()
    pruned_state_dict['bn1.bias'] = torch.mul(mask, model_state_dict['bn1.bias'][block_index]).cpu()
    pruned_state_dict['bn1.running_var'] = model_state_dict['bn1.running_var'][block_index].cpu()
    pruned_state_dict['bn1.running_mean'] = model_state_dict['bn1.running_mean'][block_index].cpu()

    for name, module in origin_model.named_modules():

        if isinstance(module, ResBasicBlock_Class):
            # conv1 & bn1
            index = torch.LongTensor(index_cfg[current_block]).to(device)
            
            pruned_weight = torch.index_select(model_state_dict[name + '.conv1.weight'], 0, index).cpu()
            pruned_weight = direct_project(pruned_weight, block_index)

            pruned_state_dict[name + '.conv1.weight'] = pruned_weight

            mask = final_mask[current_block * 2 + 1][index_cfg[current_block]]
            pruned_state_dict[name + '.bn1.weight'] = torch.mul(mask,model_state_dict[name + '.bn1.weight'][index]).cpu()
            pruned_state_dict[name + '.bn1.bias'] = torch.mul(mask,model_state_dict[name + '.bn1.bias'][index]).cpu()
            pruned_state_dict[name + '.bn1.running_var'] = model_state_dict[name + '.bn1.running_var'][index].cpu()
            pruned_state_dict[name + '.bn1.running_mean'] = model_state_dict[name + '.bn1.running_mean'][index].cpu()

            block_index = torch.LongTensor(block_index_cfg[current_block + 1]).to(device)
            mask = final_mask[current_block * 2 + 2][block_index_cfg[current_block + 1]]
            # conv2 & bn2 & shortcut
            pruned_state_dict[name + '.conv2.weight'] = torch.index_select(model_state_dict[name + '.conv2.weight'], 0, block_index).cpu()
            pruned_state_dict[name + '.conv2.weight'] = direct_project(pruned_state_dict[name + '.conv2.weight'], index)
            pruned_state_dict[name + '.bn2.weight'] = torch.mul(mask, model_state_dict[name + '.bn2.weight'][block_index]).cpu()
            pruned_state_dict[name + '.bn2.bias'] = torch.mul(mask, model_state_dict[name + '.bn2.bias'][block_index]).cpu()
            pruned_state_dict[name + '.bn2.running_var'] = model_state_dict[name + '.bn2.running_var'][block_index].cpu()
            pruned_state_dict[name + '.bn2.running_mean'] = model_state_dict[name + '.bn2.running_mean'][block_index].cpu()

            current_block += 1
    
    fc_weight = model_state_dict['fc.weight'].cpu()
    pr_fc_weight = torch.randn(fc_weight.size(0),len(block_index))
    for i, ind in enumerate(block_index):
        pr_fc_weight[:,i] = fc_weight[:,ind]
    pruned_state_dict['fc.weight'] = pr_fc_weight.cpu()
    pruned_state_dict['fc.bias'] = model_state_dict['fc.bias'].cpu()

    # load weight
    model = import_module(f'model.{args.arch}').resnet(args.cfg, layer_cfg, block_cfg).to(device)

    model.load_state_dict(pruned_state_dict)
    
    return model, [layer_cfg, block_cfg], flops, params

# Train class-wise mask   
def train_class(model, optimizer, trainLoader, epoch, topk=(1,)):

    model.train()
    losses = utils.AverageMeter(':.4e')
    accurary = utils.AverageMeter(':6.3f')
    top5_accuracy = utils.AverageMeter(':6.3f')
    print_freq = len(trainLoader.dataset) // args.train_batch_size // 10
    start_time = time.time()
    for batch, (inputs, targets) in enumerate(trainLoader):

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output, mask = model(inputs, targets)
        loss = loss_func(output, targets) 
        
        for m in mask:
            loss += float(args.sparse_lambda) * torch.sum(m, 0).norm(2)
        
        loss.backward()
        losses.update(loss.item(), inputs.size(0))
        optimizer.step()

        prec1 = utils.accuracy(output, targets, topk=topk)
        accurary.update(prec1[0], inputs.size(0))
        if len(topk) == 2:
            top5_accuracy.update(prec1[1], inputs.size(0))

        if batch % print_freq == 0 and batch != 0:
            current_time = time.time()
            cost_time = current_time - start_time
            if len(topk) == 1:
                logger.info(
                    'Epoch[{}] ({}/{}):\t'
                    'Loss {:.4f}\t'
                    'Accuracy {:.2f}%\t\t'
                    'Time {:.2f}s'.format(
                        epoch, batch * args.train_batch_size, len(trainLoader.dataset),
                        float(losses.avg), float(accurary.avg), cost_time
                    )
                )
            else:
                logger.info(
                    'Epoch[{}] ({}/{}):\t'
                    'Loss {:.4f}\t'
                    'Top1 {:.2f}%\t'
                    'Top5 {:.2f}%\t'
                    'Time {:.2f}s'.format(
                        epoch, batch * args.train_batch_size, len(trainLoader.dataset),
                        float(losses.avg), float(accurary.avg), float(top5_accuracy.avg), cost_time
                    )
                )
            start_time = current_time

# Train function
def train(model, optimizer, trainLoader, epoch, topk=(1,)):

    model.train()
    losses = utils.AverageMeter(':.4e')
    accurary = utils.AverageMeter(':6.3f')
    top5_accuracy = utils.AverageMeter(':6.3f')
    print_freq = len(trainLoader.dataset) // args.train_batch_size // 10
    start_time = time.time()
    for batch, (inputs, targets) in enumerate(trainLoader):

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_func(output, targets) 
        loss.backward()
        losses.update(loss.item(), inputs.size(0))
        optimizer.step()

        prec1 = utils.accuracy(output, targets, topk=topk)
        accurary.update(prec1[0], inputs.size(0))
        if len(topk) == 2:
            top5_accuracy.update(prec1[1], inputs.size(0))

        if batch % print_freq == 0 and batch != 0:
            current_time = time.time()
            cost_time = current_time - start_time
            if len(topk) == 1:
                logger.info(
                    'Epoch[{}] ({}/{}):\t'
                    'Loss {:.4f}\t'
                    'Accuracy {:.2f}%\t\t'
                    'Time {:.2f}s'.format(
                        epoch, batch * args.train_batch_size, len(trainLoader.dataset),
                        float(losses.avg), float(accurary.avg), cost_time
                    )
                )
            else:
                logger.info(
                    'Epoch[{}] ({}/{}):\t'
                    'Loss {:.4f}\t'
                    'Top1 {:.2f}%\t'
                    'Top5 {:.2f}%\t'
                    'Time {:.2f}s'.format(
                        epoch, batch * args.train_batch_size, len(trainLoader.dataset),
                        float(losses.avg), float(accurary.avg), float(top5_accuracy.avg), cost_time
                    )
                )
            start_time = current_time

# Test function
def test(model, testLoader, topk=(1,)):
    model.eval()

    losses = utils.AverageMeter('Time', ':6.3f')
    accurary = utils.AverageMeter('Time', ':6.3f')
    top5_accuracy = utils.AverageMeter('Time', ':6.3f')

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            predicted = utils.accuracy(outputs, targets, topk=topk)
            accurary.update(predicted[0], inputs.size(0))
            if len(topk) == 2:
                top5_accuracy.update(predicted[1], inputs.size(0))

        current_time = time.time()
        if len(topk) == 1:
            logger.info(
                'Test Loss {:.4f}\tAccuracy {:.2f}%\t\tTime {:.2f}s\n'
                .format(float(losses.avg), float(accurary.avg), (current_time - start_time))
            )
        else:
            logger.info(
                'Test Loss {:.4f}\tTop1 {:.2f}%\tTop5 {:.2f}%\tTime {:.2f}s\n'
                    .format(float(losses.avg), float(accurary.avg), float(top5_accuracy.avg), (current_time - start_time))
            )
    if len(topk) == 1:
        return accurary.avg
    else:
        return top5_accuracy.avg

def main():
    start_epoch = 0
    best_acc = 0.0
    print('==> Building Model..')
    if args.arch == 'vgg_cifar':
        origin_model = import_module(f'model.{args.arch}').VGG_Class(args.cfg).to(device)
    elif args.arch == 'resnet_cifar':
        origin_model = import_module(f'model.{args.arch}').resnet_class(args.cfg).to(device)
    elif args.arch == 'mobilenet_v2':
        origin_model = import_module(f'model.{args.arch}').MobileNetV2_class().to(device)
    else:
        raise('arch not exist!')
    print("Model Construction Down!")

    optimizer = optim.SGD(origin_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # Train the class-wise mask 
    for epoch in range(args.classtrain_epochs):
        train_class(origin_model, optimizer, loader.trainLoader, epoch, topk=(1, 5) if args.dataset == 'imagenet' else (1, ))
    print('Class_mask Traind Down!')
    
    # Global Vote for pruning
    print('==> Building Pruned Model..')
    if args.arch == 'vgg_cifar':
        model, layer_cfg, flops, params = build_vgg_pruned_model(origin_model)
    elif args.arch == 'resnet_cifar':
        model, layer_cfg, flops, params= build_resnet_pruned_model(origin_model)
    elif args.arch == 'mobilenet_v2':
        model, layer_cfg, flops, params = build_mobilenetv2_pruned_model(origin_model)
    else:
        raise('arch not exist!')

    if len(args.gpus) != 1:
        model = nn.DataParallel(model, device_ids=args.gpus)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.lr_type == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_step, gamma=0.1)
    elif args.lr_type == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs)
    print("Pruned Model Construction Down!")

    # Fine-tuning
    for epoch in range(start_epoch, args.num_epochs):
        train(model, optimizer, loader.trainLoader, epoch, topk=(1, 5) if args.dataset == 'imagenet' else (1, ))
        scheduler.step()
        test_acc = test(model, loader.testLoader, topk=(1, 5) if args.dataset == 'imagenet' else (1, ))

        is_best = best_acc < test_acc
        best_acc = max(best_acc, test_acc)

        model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()

        state = {
            'state_dict': model_state_dict,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1,
            'arch': args.cfg,
            'layer_cfg': layer_cfg
        }

        checkpoint.save_model(state, epoch + 1, is_best)

    logger.info('Pruned Model Accuracy: {:.3f}'.format(float(best_acc)))

    #Calculate Pruning rate
    orichannel = 0
    channel = 0
    for name, module in origin_model.named_modules():
        if isinstance(module, nn.Conv2d):
            orichannel += origin_model.state_dict()[name + '.weight'].size(0)

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            channel += model.state_dict()[name + '.weight'].size(0)

    logger.info('--------------UnPrune Model--------------')
    logger.info('Channels: %d'%(orichannel))
    logger.info('Params: %.2f M '%(oriparams/1000000))
    logger.info('FLOPS: %.2f M '%(oriflops/1000000))

    logger.info('--------------Prune Model--------------')
    logger.info('Channels:%d'%(channel))
    logger.info('Params: %.2f M'%(params/1000000))
    logger.info('FLOPS: %.2f M'%(flops/1000000))


    logger.info('--------------Compress Rate--------------')
    logger.info('Channels Prune Rate: %d/%d (%.2f%%)' % (channel, orichannel, 100. * (orichannel - channel) / orichannel))
    logger.info('Params Compress Rate: %.2f M/%.2f M(%.2f%%)' % (params/1000000, oriparams/1000000, 100. * (oriparams- params) / oriparams))
    logger.info('FLOPS Compress Rate: %.2f M/%.2f M(%.2f%%)' % (flops/1000000, oriflops/1000000, 100. * (oriflops- flops) / oriflops))

    logger.info('--------------Layer Configuration--------------')
    logger.info(layer_cfg)


if __name__ == '__main__':
    main()