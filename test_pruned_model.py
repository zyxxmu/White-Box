import argparse
import torch
import torch.nn as nn
from thop import profile
from importlib import import_module
import utils.common as utils

import time

from data import cifar10, imagenet

from utils.common import *


parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_set',
    type=str,
    default=None,
    help='Name of dataset. [cifar10, imagenet] default: None'
)

parser.add_argument(
    '--data_path',
    type=str,
    default=None,
    help='Path to dataset.'
)

parser.add_argument(
    '--train_batch_size',
    type=int,
    default=256,
    help='Batch size of all GPUS.'
)

parser.add_argument(
    '--eval_batch_size',
    type=int,
    default=256,
    help='Batch size of all GPUS.'
)

parser.add_argument(
    '--gpus',
    type=int,
    nargs='+',
    default=[0],
    help='Select gpu_id to use. default: [0]',
)

parser.add_argument(
    '--arch',
    type=str,
    default='resnet',
    help='Architecture of model. default:resnet'
)

parser.add_argument(
    '--cfg',
    type=str,
    default='resnet56',
    help='Detail architecuture of model. default:resnet56'
)

parser.add_argument(
    '--pruned_model_path',
    type=str,
    default=None,
    help='Path of the pruned model'
)

args = parser.parse_args()

device = torch.device(f'cuda:{args.gpus[0]}') if torch.cuda.is_available() else 'cpu'
loss_func = nn.CrossEntropyLoss()

if args.data_set == 'cifar10':
    loader = cifar10.Data(args)
elif args.data_set == 'imagenet':
    loader = imagenet.Data(args)

# Test function
if args.data_set == 'cifar10':
    def test(model, testLoader):
        global best_acc
        model.eval()
        losses = utils.AverageMeter('Loss', ':.4e')
        accuracy = utils.AverageMeter('Acc@1', ':6.2f')

        start_time = time.time()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testLoader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_func(outputs, targets)

                losses.update(loss.item(), inputs.size(0))
                pred = utils.accuracy(outputs, targets)
                accuracy.update(pred[0], inputs.size(0))

            current_time = time.time()
            print(
                f'Test Loss: {float(losses.avg):.4f}\t Acc: {float(accuracy.avg):.2f}%\t\t Time: {(current_time - start_time):.2f}s'
            )
        return accuracy.avg
elif args.data_set == 'imagenet':
    def test(model, testLoader, topk=(1,)):
        model.eval()

        losses = utils.AverageMeter('Loss', ':.4e')
        top1_accuracy = utils.AverageMeter('Acc@1', ':6.2f')
        top5_accuracy = utils.AverageMeter('Acc@5', ':6.2f')

        start_time = time.time()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testLoader):
                inputs, targets = inputs.to(device), targets.to(device)

                # compute output
                outputs = model(inputs)
                loss = loss_func(outputs, targets)

                # measure accuracy and record loss
                losses.update(loss.item(), inputs.size(0))
                pred = utils.accuracy(outputs, targets, topk=topk)
                top1_accuracy.update(pred[0], inputs.size(0))
                top5_accuracy.update(pred[1], inputs.size(0))

            # measure elapsed time
            current_time = time.time()
            print(
                f'Test Loss: {float(losses.avg):.6f}\t Top1: {float(top1_accuracy.avg):.6f}%\t'
                f'Top5: {float(top5_accuracy.avg):.6f}%\t Time: {float(current_time - start_time):.2f}s'
            )

        return float(top1_accuracy.avg), float(top5_accuracy.avg)

if args.data_set == 'cifar10':
    resumeckpt = torch.load(f'{args.pruned_model_path}')
    state_dict = resumeckpt['state_dict']
    cfg = resumeckpt['layer_cfg']
    if args.arch == 'vgg_cifar':
        model = import_module(f'model.{args.arch}').VGG(args.cfg, layer_cfg=cfg).to(device)
    elif args.arch == 'resnet_cifar':
        model = import_module(f'model.{args.arch}').resnet(args.cfg, layer_cfg=cfg[0], block_cfg=cfg[1]).to(device)
    elif args.arch == 'mobilenet_v2':
        model = import_module(f'model.{args.arch}').MobileNetV2(wm=1,layer_cfg=cfg).to(device)
    else:
        raise('arch not exist!')

elif args.data_set == 'imagenet':
    resumeckpt = torch.load(f'{args.pruned_model_path}')
    state_dict = resumeckpt['state_dict']
    cfg = resumeckpt['cfg']
    if args.arch == 'resnet':
        model = import_module(f'model.{args.arch}').resnet(args.cfg, block_cfg = cfg[1],layer_cfg=cfg[0]).to(device)
    else:
        raise('arch not exist!')
model.load_state_dict(state_dict)

if args.arch == 'vgg_cifar':
    origin_model = import_module(f'model.{args.arch}').VGG(args.cfg).to(device)
elif args.arch == 'resnet_cifar':
    origin_model = import_module(f'model.{args.arch}').resnet(args.cfg).to(device)
elif args.arch == 'mobilenet_v2':
    origin_model = import_module(f'model.{args.arch}').MobileNetV2(wm=1).to(device)
elif args.arch == 'resnet':
    origin_model = import_module(f'model.{args.arch}').resnet(args.cfg).to(device)

if len(args.gpus) != 1:
    model = nn.DataParallel(model, device_ids=args.gpus)

if args.data_set == 'cifar10':
    compact_test_acc = float(test(model, loader.testLoader))
    print(f'Best Compact Model Acc Top1: {compact_test_acc:.2f}%')
    inputs = torch.randn(1, 3, 32, 32).to(device)

elif args.data_set == 'imagenet':
    compact_test_acc = test(model, loader.testLoader, topk=(1, 5))
    print(f'Best Compact Model Acc Top1: {compact_test_acc[0]:.2f}%, Top5: {compact_test_acc[1]:.2f}%')
    inputs = torch.randn(1, 3, 224, 224).to(device)

oriflops, oriparams = profile(origin_model, inputs=(inputs, ))
flops, params = profile(model, inputs=(inputs, ))

print('--------------UnPrune Model--------------')
print('Params: %.2f M '%(oriparams/1000000))
print('FLOPS: %.2f M '%(oriflops/1000000))

print('--------------Prune Model--------------')
print('Params: %.2f M'%(params/1000000))
print('FLOPS: %.2f M'%(flops/1000000))

print('--------------Compress Rate--------------')
print('Params Compress Rate: %.2f M/%.2f M(%.2f%%)' % (params/1000000, oriparams/1000000, 100. * (oriparams- params) / oriparams))
print('FLOPS Compress Rate: %.2f M/%.2f M(%.2f%%)' % (flops/1000000, oriflops/1000000, 100. * (oriflops- flops) / oriflops))
