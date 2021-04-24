import torch
import torch.nn as nn
import torch.nn.functional as F


norm_mean, norm_var = 0.0, 1.0

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ClassScale(torch.autograd.Function):
    '''
    input: x: 64*512*7*7, scale:512 ==> x[:, i, :, :]*scale[i]
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
        guide = torch.index_select(mask, 0, targets)
        self.save_for_backward(x, mask, guide, targets)
        guide = torch.squeeze(guide.reshape(-1))
        
        out = torch.mul(x_1, guide)
        out = out.transpose(0, 1)
        out = out.reshape(n, c, h, w)
        return out

    @staticmethod
    def backward(self, grad_output):
        x, mask, guide, targets = self.saved_tensors
        #x: N x C x h x w
        #mask: num_classes x C 
        #targets: N
        #guide: N x C
        #grade_output.data: N x C x h x w

        #grade_x: N x C x h x w
        #grade_mask: num_classes x C

        grad_guide = guide.clone()
        grad = grad_output.data.clone()
        n, c, h, w = grad.size()

        grad = grad.reshape(n*c,h*w).transpose(0,1)
        grad_x = grad.clone()
        grad_guide = torch.squeeze(grad_guide.reshape(-1))
        
        #grade
        grad_x = torch.mul(grad, grad_guide)
        grad_x = grad_x.transpose(0, 1)
        grad_x = grad_x.reshape(n, c, h, w)

        #grad_guide 
        x = x.reshape(n*c,h*w).transpose(0,1)
        grad_guide = guide.clone()
        grad_guide = torch.sum((grad*x).transpose(0,1),1)
        grad_guide = grad_guide.reshape(n, c)

        grad_mask = torch.zeros_like(mask)

        for i in range(grad_mask.shape[0]):
            grad_mask[i] = torch.sum(torch.mul(torch.unsqueeze(torch.eq(targets, i), 1), grad_guide),0)
        
        return grad_x, grad_mask, None

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

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class ResBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, filter_num, stride=1):
        super(ResBasicBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = conv3x3(inplanes, filter_num, stride)
        self.bn1 = nn.BatchNorm2d(filter_num)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(filter_num, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            if stride!=1:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2],
                                    (0, 0, 0, 0, (planes-inplanes)//2, planes-inplanes-(planes-inplanes)//2), "constant", 0))
            else:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, :, :],
                                    (0, 0, 0, 0, (planes-inplanes)//2, planes-inplanes-(planes-inplanes)//2), "constant", 0))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out

class ResBasicBlock_Class(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, filter_num, stride=1):
        super(ResBasicBlock_Class, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = conv3x3(inplanes, filter_num, stride)
        self.bn1 = nn.BatchNorm2d(filter_num)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(filter_num, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            if stride!=1:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2],
                                    (0, 0, 0, 0, (planes-inplanes)//2, planes-inplanes-(planes-inplanes)//2), "constant", 0))
            else:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, :, :],
                                    (0, 0, 0, 0, (planes-inplanes)//2, planes-inplanes-(planes-inplanes)//2), "constant", 0))

    def forward(self, x, mask, mask_1, targets):
        out = self.conv1(x)
        out = ClassScale_Gaussian.apply(out, mask, targets)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)


        return out

class ResNet_class(nn.Module):
    def __init__(self, block, num_layers, layer_cfg=None, block_cfg=None, num_classes=10):
        super(ResNet_class, self).__init__()
        block_layers = num_layers - 2
        assert block_layers % 6 == 0, 'depth should be 6n+2'
        n = block_layers // 6

        block_cfg = [16, 32, 64]

        self.layer_cfg = layer_cfg
        self.cfg_index = 0
        self.inplanes = block_cfg[0]
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, block_cfg[0], blocks=n, stride=1)
        self.layer2 = self._make_layer(block, block_cfg[1], blocks=n, stride=2)
        self.layer3 = self._make_layer(block, block_cfg[2], blocks=n, stride=2)

        self.mask = nn.ParameterList([nn.Parameter(torch.ones(num_classes, block_cfg[int(i / (block_layers // 3))])) for i in range(-1, block_layers)])
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(block.expansion * block_cfg[2], num_classes)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride):
        layers = []

        layers.append(block(self.inplanes, planes, filter_num=
                    self.layer_cfg[self.cfg_index] if self.layer_cfg != None else planes,
                                        stride=stride))
        self.cfg_index += 1

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, filter_num=
                    self.layer_cfg[self.cfg_index] if self.layer_cfg != None else planes))
            self.cfg_index += 1

        return nn.Sequential(*layers)

    def forward(self, x, targets):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        mask_index = 1 
        for block in self.layer1:
            x = block(x, self.mask[mask_index], self.mask[mask_index + 1], targets)
            mask_index += 2
        for block in self.layer2:
            x = block(x, self.mask[mask_index], self.mask[mask_index + 1], targets)
            mask_index += 2
        for block in self.layer3:
            x = block(x, self.mask[mask_index], self.mask[mask_index + 1], targets)
            mask_index += 2
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, self.mask

class ResNet(nn.Module):
    def __init__(self, block, num_layers, layer_cfg=None, block_cfg=None, num_classes=10):
        super(ResNet, self).__init__()
        assert (num_layers - 2) % 6 == 0, 'depth should be 6n+2'
        n = (num_layers - 2) // 6

        if block_cfg == None:
            block_cfg = [16, 32, 64]
        self.layer_cfg = layer_cfg
        self.cfg_index = 0
        self.inplanes = block_cfg[0]
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, block_cfg[0], blocks=n, stride=1)
        self.layer2 = self._make_layer(block, block_cfg[1], blocks=n, stride=2)
        self.layer3 = self._make_layer(block, block_cfg[2], blocks=n, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(block_cfg[2] * block.expansion, num_classes)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride):
        layers = []

        layers.append(block(self.inplanes, planes, filter_num=
                    self.layer_cfg[self.cfg_index] if self.layer_cfg != None else planes,
                                        stride=stride))
        self.cfg_index += 1

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, filter_num=
                    self.layer_cfg[self.cfg_index] if self.layer_cfg != None else planes))
            self.cfg_index += 1

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet56(layer_cfg=None, block_cfg = None, **kwargs):
    return ResNet(ResBasicBlock, 56, layer_cfg, block_cfg, **kwargs)

def resnet110(layer_cfg=None,block_cfg = None,  **kwargs):
    return ResNet(ResBasicBlock, 110, layer_cfg, block_cfg, **kwargs)

def resnet(arch, layer_cfg=None, block_cfg=None,**kwargs):
    if arch == 'resnet56':
        return resnet56(layer_cfg, block_cfg, **kwargs)
    elif arch == 'resnet110':
        return resnet110(layer_cfg, block_cfg, **kwargs)

def resnet56_class(layer_cfg=None, block_cfg = None, **kwargs):
    return ResNet_class(ResBasicBlock_Class, 56, layer_cfg, block_cfg, **kwargs)

def resnet110_class(layer_cfg=None,block_cfg = None,  **kwargs):
    return ResNet_class(ResBasicBlock_Class, 110, layer_cfg, block_cfg, **kwargs)

def resnet_class(arch, layer_cfg=None, block_cfg=None,**kwargs):
    if arch == 'resnet56':
        return resnet56_class(layer_cfg, block_cfg, **kwargs)
    elif arch == 'resnet110':
        return resnet110_class(layer_cfg, block_cfg, **kwargs)