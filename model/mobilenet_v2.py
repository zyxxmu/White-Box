'''MobileNetV2 in PyTorch.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def Gaussian_select(mask, targets):
    '''
    #output:  distribution:N*num_classes
    #mask: num_classes x C
    #guide: N*C
    '''

    distri = torch.rand(targets.size(0), mask.size(0),device=mask.device)

    for i, j in enumerate(targets):
        distri[i][j] = 1

    guide = torch.mm(distri, mask)

    return guide, distri

class ClassScale_Gaussian(torch.autograd.Function):
    '''
    input: x: 64*512*7*7, mask:10*512, scale: mask[target] * 1 + mask[other] * Gaussian_distribution==> x[:, i, :, :]*scale[i]
    '''

    @staticmethod
    def forward(self, x, mask, targets):
        #x: N x C x h x w 
        #mask: num_classes x C
        #targets: N
        #guide: N x C
        #output: N x C x h x W

        n, c, h, w = x.size()
        x_1 = x.reshape(n*c,h*w)
        x_1 = x_1.transpose(0,1)

        guide, distri = Gaussian_select(mask, targets)
        self.save_for_backward(x, mask, guide, distri, targets)
        guide = torch.squeeze(guide.reshape(-1))
        
        out = torch.mul(x_1, guide)
        out = out.transpose(0, 1)
        out = out.reshape(n, c, h, w)
        #self.save_for_backward(input_data, scale_vec)
        return out

    @staticmethod
    def backward(self, grad_output):
        x, mask, guide, distri, targets = self.saved_tensors
        #x: N x C x h x w
        #mask: num_classes x C 
        #targets: N
        #guide: N x C
        #grade_output.data: N x C x h x w
        #distri: N*num_classes

        #grade_x: N x C x h x w
        #grade_mask: num_classes x C

        guide_1 = guide.clone()
        grad = grad_output.data.clone()
        n, c, h, w = grad.size()

        grad = grad.reshape(n*c,h*w).transpose(0,1)
        grad_x = grad.clone()
        guide_1 = torch.squeeze(guide_1.reshape(-1))
        
        #grade
        grad_x = torch.mul(grad, guide_1)
        grad_x = grad_x.transpose(0, 1)
        grad_x = grad_x.reshape(n, c, h, w)

        #grad_guide 
        x = x.reshape(n*c,h*w).transpose(0,1)
        grad_guide = guide.clone()
        grad_guide = torch.sum((grad*x).transpose(0,1),1)
        grad_guide = grad_guide.reshape(n, c)

        grad_mask = torch.zeros_like(mask)

        for i in range(grad_mask.shape[0]):
            grad_mask[i] = torch.sum(torch.mul(torch.unsqueeze(distri[:,i],1), grad_guide),0)
        
        return grad_x, grad_mask, None 

class Block_class(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block_class, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x, mask, targets):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = ClassScale_Gaussian.apply(out, mask, targets)
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2_class(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 1, 1],
        [6, 24, 1, 1],
        [6, 32, 1, 2],
        [6, 32, 1, 1],
        [6, 32, 1, 1],
        [6, 64, 1, 2],
        [6, 64, 1, 1],
        [6, 64, 1, 1],
        [6, 64, 1, 1],
        [6, 96, 1, 1],
        [6, 96, 1, 1],
        [6, 96, 1, 1],
        [6, 160, 1, 2],
        [6, 160, 1, 1],
        [6, 160, 1, 1],
        [6, 320, 1, 1],
    ]


    def __init__(self, num_classes=10, layer_cfg = None):
        super(MobileNetV2_class, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        mask_cfg = [32, 96, 144, 144, 192, 192, 192, 384, 384, 384, 384, 576, 576, 576, 960, 960, 960]
        self.layer_cfg = layer_cfg
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)
        self.mask = nn.ParameterList([nn.Parameter(torch.ones(num_classes, mask_cfg[i])) for i in range(len(mask_cfg))])

    def _make_layers(self, in_planes):
        layers = []
        for i, (expansion, out_planes, num_blocks, stride) in enumerate(self.cfg):
            layers.append(Block_class(in_planes, out_planes, expansion, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x, targets):
        out = F.relu(self.bn1(self.conv1(x)))
        #   out = self.layers(out)
        index = 0
        for block in self.layers:
            out = block(out, self.mask[index], targets)
            index += 1
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, self.mask



class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride, planes=None):
        super(Block, self).__init__()
        self.stride = stride

        if planes ==  None:
            planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 1, 1],
        [6, 24, 1, 1],
        [6, 32, 1, 1],
        [6, 32, 1, 1],
        [6, 32, 1, 1],
        [6, 64, 1, 2],
        [6, 64, 1, 1],
        [6, 64, 1, 1],
        [6, 64, 1, 1],
        [6, 96, 1, 1],
        [6, 96, 1, 1],
        [6, 96, 1, 1],
        [6, 160, 1, 2],
        [6, 160, 1, 1],
        [6, 160, 1, 1],
        [6, 320, 1, 1],
    ]

    def __init__(self, wm=1,num_classes=10, layer_cfg = None):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.wm = wm
        if layer_cfg == None:
            layer_cfg = [32, 96, 144, 144, 192, 192, 192, 384, 384, 384, 384, 576, 576, 576, 960, 960, 960, 1280]
        self.layer_cfg = layer_cfg
        self.conv1 = nn.Conv2d(3, int(32*self.wm), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(32*self.wm))
        self.layers = self._make_layers(in_planes=int(32*self.wm))
        self.conv2 = nn.Conv2d(int(320*self.wm), int(layer_cfg[-1]*self.wm), kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(int(layer_cfg[-1]*self.wm))
        self.linear = nn.Linear(int(layer_cfg[-1]*self.wm), num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for i, (expansion, out_planes, num_blocks, stride) in enumerate(self.cfg):
            out_planes = int(out_planes*self.wm)
            layers.append(Block(in_planes, out_planes, expansion, stride, self.layer_cfg[i]))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)), inplace = True)
        out = self.layers(out)
        out = F.relu6(self.bn2(self.conv2(out)), inplace = True)
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        #out = F.avg_pool2d(out, 4)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    import thop
    from thop import profile
    net = MobileNetV2(wm=1)
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())
    flops, params = profile(net, inputs=(x, ))
    print('--------------UnPrune Model--------------')
    print('Params: %.2f M '%(params/1000000))
    print('FLOPS: %.2f M '%(flops/1000000))
#test()