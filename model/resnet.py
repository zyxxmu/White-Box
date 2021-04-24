import torch
import torch.nn as nn
import torch.nn.functional as F


conv_num_cfg = {
    'resnet18' : 12,
    'resnet34' : 20,
    'resnet50' : 20,
    'resnet101' : 37,
    'resnet152' : 54 
}

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

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, ori_planes, planes, honey, index, stride=1):
        super(BasicBlock, self).__init__()
        pr_channels = int(ori_planes * honey[index] / 10)
        self.conv1 = nn.Conv2d(in_planes, pr_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(pr_channels)
        self.conv2 = nn.Conv2d(pr_channels, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

class Bottleneck_class(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_class, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x, mask_0, mask_1, mask_2, targets):
        out = F.relu(self.bn1(self.conv1(x)))
        out = ClassScale_Gaussian.apply(out, mask_0, targets)
        out = F.relu(self.bn2(self.conv2(out)))
        out = ClassScale_Gaussian.apply(out, mask_1, targets)
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

class ResNet_class(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet_class, self).__init__()
        self.in_planes = 64
        self.num_blocks = num_blocks

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,  bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layers(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layers(block, 128, num_blocks[1],  stride=2)
        self.layer3 = self._make_layers(block, 256, num_blocks[2],  stride=2)
        self.layer4 = self._make_layers(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.Sequential(nn.AvgPool2d(7))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        mask_cfg = [64, 64, 256, 64, 64, 256, 64, 64, 256,
                    128, 128, 512, 128, 128, 512, 128, 128, 512, 128, 128, 512,
                    256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024,
                    512, 512, 2048, 512, 512, 2048, 512, 512, 2048]

        self.mask = nn.ParameterList([nn.Parameter(torch.ones(num_classes, mask_cfg[i])) for i in range(len(mask_cfg))])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layers(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, targets):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        index = 0
        for feature in self.layer1:
            out = feature(out, self.mask[index], self.mask[index+1], self.mask[index+2], targets)
            index += 3
        for feature in self.layer2:
            out = feature(out, self.mask[index], self.mask[index+1], self.mask[index+2], targets)
            index += 3
        for feature in self.layer3:
            out = feature(out, self.mask[index], self.mask[index+1], self.mask[index+2], targets)
            index += 3
        for feature in self.layer4:
            out = feature(out, self.mask[index], self.mask[index+1], self.mask[index+2], targets)
            index += 3
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out, self.mask

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, pr_channels_1, pr_channels_2, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, pr_channels_1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(pr_channels_1)
        self.conv2 = nn.Conv2d(pr_channels_1, pr_channels_2,kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(pr_channels_2)
        self.conv3 = nn.Conv2d(pr_channels_2, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, block_cfg, layer_cfg, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.layer_cfg = layer_cfg
        self.current_conv = 0

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,  bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layers(block, block_cfg[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layers(block, block_cfg[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layers(block, block_cfg[2], num_blocks[2],  stride=2)
        self.layer4 = self._make_layers(block, block_cfg[3], num_blocks[3], stride=2)

        self.avgpool = nn.Sequential(nn.AvgPool2d(7))
        self.fc = nn.Linear(block_cfg[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes,
                self.layer_cfg[self.current_conv], self.layer_cfg[self.current_conv+1], stride))
            self.current_conv +=2
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def resnet(cfg, block_cfg = None, layer_cfg = None, num_classes = 1000):
    if block_cfg == None:
        block_cfg = [64, 128, 256, 512]
    if layer_cfg == None:
        layer_cfg = [64, 64, 64, 64, 64, 64,
                    128, 128, 128, 128, 128, 128, 128, 128, 
                    256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
                    512, 512, 512, 512, 512, 512]
    if cfg == 'resnet18':
        return ResNet(BasicBlock, [2,2,2,2], block_cfg = block_cfg, layer_cfg = layer_cfg, num_classes=num_classes)
    elif cfg == 'resnet34':
        return ResNet(BasicBlock, [3,4,6,3], block_cfg = block_cfg, layer_cfg = layer_cfg, num_classes=num_classes)
    elif cfg == 'resnet50':
        return ResNet(Bottleneck, [3,4,6,3], block_cfg = block_cfg, layer_cfg = layer_cfg, num_classes=num_classes)
    elif cfg == 'resnet101':
        return ResNet(Bottleneck, [3,4,23,3], block_cfg = block_cfg, layer_cfg = layer_cfg, num_classes=num_classes)
    elif cfg == 'resnet152':
        return ResNet(Bottleneck, [3,8,36,3], block_cfg = block_cfg, layer_cfg = layer_cfg, num_classes=num_classes)

def resnet_class(cfg, num_classes = 1000):
    if cfg == 'resnet50':
        return ResNet_class(Bottleneck_class, [3,4,6,3], num_classes=num_classes)

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    if block_cfg == None:
        block_cfg = [64, 128, 256, 512]
    if layer_cfg == None:
        layer_cfg = [64, 64, 64, 64, 64, 64,
                    128, 128, 128, 128, 128, 128, 128, 128, 
                    256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
                    512, 512, 512, 512, 512, 512]
    return ResNet(Bottleneck, [3,4,6,3], block_cfg = block_cfg, layer_cfg = layer_cfg, num_classes =1000)

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])
