# Carrying out CNN Channel Pruning in a White Box ([Paper Link]()) ![](https://visitor-badge.glitch.me/badge?page_id=zyxxmu.White-Box)
## Requirements

* python3.7.4, pytorch 1.5.1, torchvision 0.4.2, thop 0.0.31

## Reproduce the Experiment Results 

Run the following scripts to reproduce the results reported in paper (change your data path in the corresponding scripts).

* VGGNet-16-CIFAR10 ./scripts/vgg.sh
* ResNet-56-CIFAR10 ./scripts/resnet56.sh   
* ResNet-110-CIFAR10 ./scripts/resnet110.sh 
* MobileNet-v2-CIFAR10 ./scripts/mobilenetv2.sh  
* ResNet-50-ImageNet(FLOPs:2.22B) ./scripts/resnet50-1.sh  
* ResNet-50-ImageNet(FLOPs:1.50B) ./scripts/resnet50-2.sh  

## Evaluate Our Pruned Models

Run the following scripts to test our results reported in the paper (change your data path and input the pruned model path in the corresponding scripts. The pruned moodel can be downloaded from the links in the following table).

* VGGNet-16-CIFAR10 ./scripts/test-vgg.sh
* ResNet-56-CIFAR10 ./scripts/test-resnet56.sh   
* ResNet-110-CIFAR10 ./scripts/test-resnet110.sh 
* MobileNet-v2-CIFAR10 ./scripts/test-mobilenetv2.sh  
* ResNet-50-ImageNet(FLOPs:2.22B) ./scripts/test-resnet50-1.sh  
* ResNet-50-ImageNet(FLOPs:1.50B) ./scripts/test-resnet50-2.sh  

#### CIFAR-10

| Full Model   | Flops &#8595; | Accuracy | Pruned Model                                                 |
| ------------ | ----------------- | -------- | ------------------------------------------------------------ |
| VGG16        | 76.4%             | 93.47%   | [Modellink](https://drive.google.com/drive/folders/1GWR56Aoc08r3eUUwSub1_lxJ0Z06dWyd?usp=sharing) |
| ResNet56     | 55.6%             | 93.54%   | [Modellink](https://drive.google.com/drive/folders/1NSnJnLGWsSJLiVCksk1OnOK2iVGRfLyg?usp=sharing) |
| ResNet110    | 66.0%             | 94.12%   | [Modellink](https://drive.google.com/drive/folders/1h-eSUbtJ_xO3wlnQ7J3Pl8bBsuTEw9LJ?usp=sharing) |
| MobileNet-V2 | 29.2%             | 95.28%   | [Modellink](https://drive.google.com/drive/folders/1Q78kM5U8Tz-nonCLbBisVrke97OGIIai?usp=sharing) |

#### ImageNet

| Network  | Flops &#8595; | Top-1 Acc. | Top-5 Acc.  | Pruned Model                                                 |
| -------- | ----------------- | -------- | -------- | ------------------------------------------------------------ |
| ResNet50 | 45.6%             | 75.32%   | 92.43%   | [Modellink](https://drive.google.com/drive/folders/1WGWce2puviqwKfjWrxB9CLJcotJxOx_a?usp=sharing) |
| ResNet50 | 63.5%             | 74.21%   | 92.01%   | [Modellink](https://drive.google.com/drive/folders/15C6RvrLvPoswrXKpvT_idCm8rB88zxLB?usp=sharing) |

