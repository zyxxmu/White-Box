import argparse

parser = argparse.ArgumentParser(description='White-Box')

parser.add_argument(
    '--gpus',
    type=int,
    nargs='+',
    default=[0],
    help='Select gpu_id to use. default:[0]',
)

parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    help='Select dataset to train. default:cifar10',
)

parser.add_argument(
    '--data_path',
    type=str,
    default='/data/cifar10/',
    help='The dictionary where the input is stored. default:/data/cifar10/',
)

parser.add_argument(
    '--job_dir',
    type=str,
    default='experiments/',
    help='The directory where the summaries will be stored. default:./experiments'
)

parser.add_argument(
    '--resume',
    action='store_true',
    help='Load the model from the specified checkpoint.'
)

## Training
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
    '--num_epochs',
    type=int,
    default=300,
    help='The number of epoch to train. default:300'
)

parser.add_argument(
    '--train_batch_size',
    type=int,
    default=256,
    help='Batch size for training. default:256'
)

parser.add_argument(
    '--eval_batch_size',
    type=int,
    default=100,
    help='Batch size for validation. default:100'
)

parser.add_argument(
    '--momentum',
    type=float,
    default=0.9,
    help='Momentum for MomentumOptimizer. default:0.9'
)

parser.add_argument(
    '--lr',
    type=float,
    default=1e-2,
    help='Learning rate for train. default:1e-2'
)

parser.add_argument(
    '--lr_type',
    default='step', 
    type=str, 
    help='lr scheduler (step/exp/cos/step3/fixed)'
)

parser.add_argument(
    '--criterion',
    default='Softmax', 
    type=str, 
    help='Loss func (Softmax)'
)

parser.add_argument(
    '--lr_decay_step',
    type=int,
    nargs='+',
    default=[50, 100],
    help='the iterval of learn rate. default:50, 100'
)

parser.add_argument(
    '--weight_decay',
    type=float,
    default=5e-3,
    help='The weight decay of loss. default:5e-3'
)

parser.add_argument(
    '--pruning_rate',
    type=float,
    default=0.5,
    help='Target Pruning Rate. default:0.5'
)

parser.add_argument(
    '--classtrain_epochs',
    type=int,
    default=30,
    help='Train_class_epochs'
)

parser.add_argument(
    '--sparse_lambda',
    type=float,
    default=0.0001,
    help='Sparse_lambda. default:0.00001'
)

parser.add_argument(
    '--min_preserve',
    type=float,
    default=0.3,
    help='Minimum preserve percentage of each layer. default:0.3'
)

parser.add_argument(
    '--debug',
    action='store_true',
    help='input to open debug state'
)

args = parser.parse_args()