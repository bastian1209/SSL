import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from copy import deepcopy


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, padding=1, norm_layer=None, is_last=False):
        super(BasicBlock, self).__init__()
        
        self.is_last = is_last
        self.norm_layer = nn.BatchNorm2d if norm_layer == None else norm_layer
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = self.norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = self.norm_layer(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                self.norm_layer(self.expansion * planes)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        
        if self.is_last:
            return out, preact
        else:
            return out
        
        
class BottleNeck(nn.Module):
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1, norm_layer=None, is_last=False):
        super(BottleNeck, self).__init__()
        self.is_last = is_last
        self.norm_layer = nn.BatchNorm2d if norm_layer == None else norm_layer
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = self.norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = self.norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = self.norm_layer(self.expansion * planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                self.norm_layer(self.expansion * planes)
            )
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        
        if self.is_last:
            return out, preact
        else:
            return out
        
class BasicBlock_wo_BN(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, padding=1, norm_layer=None, is_last=False):
        super(BasicBlock_wo_BN, self).__init__()
        self.is_last = is_last
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
            )
            
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        
        if self.is_last:
            return out, preact
        else:
            return out
        
        
class BottleNeck_wo_BN(nn.Module):
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1, norm_layer=None, is_last=False):
        super(BottleNeck_wo_BN, self).__init__()
        self.is_last = is_last
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
            )
        
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        out += self.shortcut(x)
        preact = out
        
        if self.is_last:
            return out, preact
        else:
            return out
        
            
class ResNet(nn.Module):
    def __init__(self, block, num_blocks_list, in_channels=3, norm_layer=None, is_cifar=False, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.norm_layer = nn.BatchNorm2d if norm_layer == None else norm_layer
        
        if is_cifar:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False), 
                self.norm_layer(self.in_planes), 
                nn.ReLU(inplace=True)
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False), 
                self.norm_layer(self.in_planes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        
        self.layer1 = self._make_layer(block, 64, num_blocks_list[0])
        self.layer2 = self._make_layer(block, 128, num_blocks_list[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks_list[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks_list[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
        
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn3.weight, 0.)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0.)
        
    def _make_layer(self, block, planes, num_blocks, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            layers.append(block(self.in_planes, planes, strides[i], self.norm_layer))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        
        return out
    

class ResNet_wo_BN(nn.Module):
    def __init__(self, block, num_blocks_list, in_channels=3, norm_layer=None, is_cifar=False, zero_init_residual=False):
        super(ResNet_wo_BN, self).__init__()
        self.in_planes = 64
        
        if is_cifar:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False), 
                nn.ReLU(inplace=True)
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False), 
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        
        self.layer1 = self._make_layer(block, 64, num_blocks_list[0])
        self.layer2 = self._make_layer(block, 128, num_blocks_list[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks_list[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks_list[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
        
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn3.weight, 0.)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0.)
        
    def _make_layer(self, block, planes, num_blocks, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            layers.append(block(self.in_planes, planes, strides[i], None))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        
        return out
    
    
def resnet18(**kwargs):
    if kwargs["BN"]:
        kwargs.pop("BN")
        return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    else:
        kwargs.pop("BN")
        return ResNet_wo_BN(BasicBlock_wo_BN, [2, 2, 2, 2], **kwargs)


def resnet50(**kwargs):
    if kwargs["BN"]:
        kwargs.pop("BN")
        return ResNet(BottleNeck, [3, 4, 6, 3], **kwargs)
    else:
        kwargs.pop("BN")
        return ResNet_wo_BN(BottleNeck_wo_BN, [3, 4, 6, 3], **kwargs)


model_dict = {
    'resnet18' : [resnet18, 512], 
    'resnet50' : [resnet50, 2048]
}


class MLPhead(nn.Module):
    def __init__(self, in_features, out_features, hidden, bn_mlp):
        super(MLPhead, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        if bn_mlp[0]:
            self.bn1 = nn.BatchNorm1d(hidden)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, out_features)
        if bn_mlp[1]:
            self.bn2 = nn.BatchNorm1d(out_features)
            
    def forward(self, x):
        x = self.fc1(x)
        if hasattr(self, 'bn1'):
            x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if hasattr(self, 'bn2'):
            x = self.bn2(x)
        
        return x
    
    
class MLPhead3(nn.Module):
    def __init__(self, in_features, out_features, hiddens, bn_mlp):
        super(MLPhead3, self).__init__()
        self.fc1 = nn.Linear(in_features, hiddens[0])
        if bn_mlp[0]:
            self.bn1 = nn.BatchNorm1d(hiddens[0])
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hiddens[0], hiddens[1])
        if bn_mlp[1]:
            self.bn2 = nn.BatchNorm1d(hiddens[1])
        self.fc3 = nn.Linear(hiddens[1], out_features)
        if bn_mlp[2]:
            self.bn3 = nn.BatchNorm1d(out_features)
            
    def forward(self, x):
        x = self.fc1(x)
        if hasattr(self, 'bn1'):
            x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if hasattr(self, 'bn2'):
            x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        if hasattr(self, 'bn3'):
            x = self.bn3(x)
            
        return x
    
    
class LensNet(nn.Module):
    def __init__(self, block, num_blocks=4, num_channels=64):
        super(LensNet, self).__init__()
        self.num_blocks = num_blocks
        self.num_channels = num_channels
        block.expansion = 1
        
        encoder_modules = []
        for i in range(num_blocks):
            if i == 0:
                encoder_modules.append(block(3, num_channels))
            else:
                in_planes = int(2 ** (i - 1) * num_channels)
                planes = int(2 ** i * num_channels)
                encoder_modules.append(block(in_planes, planes)) 
        self.encoder = nn.Sequential(*encoder_modules)
        
        decoder_modules = []
        for i in range(num_blocks):
            if i == self.num_blocks - 1:
                in_planes = 2 * num_channels
                planes = num_channels
            else:    
                in_planes = int(2 * 2 ** (num_blocks - (i + 1)) * num_channels)
                planes = int(2 ** (num_blocks - (i + 2)) * num_channels)
            decoder_modules.append(block(in_planes, planes))
        self.decoder = nn.Sequential(*decoder_modules)
        
        self.bottleneck = nn.Sequential(
            block(int(2 ** (num_blocks - 1) * num_channels), int(2 ** num_blocks * num_channels)),
            block(int(2 ** num_blocks * num_channels), int(2 ** (num_blocks - 1) * num_channels))
        )
        
        self.pre_logit = nn.Conv2d(num_channels, 3, 1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        
    def forward(self, x):
        x_img = deepcopy(x)
        encoder_outs = []
        for encoder_module in self.encoder:
            x = encoder_module(x)
            encoder_outs.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        
        for decoder_module, encoder_out in zip(self.decoder, encoder_outs[::-1]):
            x = self.upsample(x)
            x = torch.cat([encoder_out, x], dim=1)
            x = decoder_module(x)
            
        x = self.pre_logit(x)
            
        return x + x_img
    
    
class ResNet_SSL(nn.Module):
    def __init__(self, arch_name='resnet50', head='mlp', 
                 encoder_params={"BN" : True, "norm_layer" : None, 'is_cifar' : False, 'zero_init_residual' : False}, ssl_feat_dim=128, hidden_double=False, bn_mlp=[False, False]):
        super(ResNet_SSL, self).__init__()
        model_func, feat_dim = model_dict[arch_name]
        self.encoder = model_func(**encoder_params)
        
        if head == 'mlp':
            if hidden_double:
                hidden = 2 * feat_dim
            else:
                hidden = feat_dim
            self.proj_head = MLPhead(feat_dim, ssl_feat_dim, hidden=hidden, bn_mlp=bn_mlp)
            # self.proj_head = MLPhead(feat_dim, ssl_feat_dim, hidden=ssl_feat_dim, bn_mlp=bn_mlp)
        elif head == 'linear':
            self.proj_head = nn.Linear(feat_dim, ssl_feat_dim)
            
    def forward(self, x):
        feature = self.encoder(x)
        feature = F.normalize(self.proj_head(feature), dim=1)
        
        return feature
    

class ResNet_SimSiam(nn.Module):     
    def __init__(self, arch_name='resnet50', head='mlp', 
                 encoder_params={"BN" : True, "norm_layer" : None, 'is_cifar' : False, 'zero_init_residual' : False}, ssl_feat_dim=2048, hidden_double=False, bn_mlp=[True, True], regular_pred=False):
        super(ResNet_SimSiam, self).__init__()
        model_func, feat_dim = model_dict[arch_name]
        self.encoder = model_func(**encoder_params)
        
        hiddens = [feat_dim, feat_dim]
        if head == 'mlp':
            if hidden_double:
                hidden[1] = 2 * feat_dim
            if encoder_params['is_cifar']:
                # this should be commented out
                if regular_pred:
                    self.proj_head = MLPhead(feat_dim, ssl_feat_dim, hidden=hiddens[0], bn_mlp=bn_mlp)
                else:
                    self.proj_head = MLPhead(feat_dim, ssl_feat_dim, hidden=ssl_feat_dim, bn_mlp=bn_mlp)
            else:
                self.proj_head = MLPhead3(feat_dim, ssl_feat_dim, hiddens=hiddens, bn_mlp=bn_mlp + [True])
        
    def forward(self, x):
        feature = self.encoder(x)
        feature = F.normalize(self.proj_head(feature), dim=1)
        
        return feature
    
    
class ResNet_SL(nn.Module):
    def __init__(self, arch_name='resnet50', encoder_params={"BN" : True, "norm_layer" : None, "is_cifar" : False}, num_classes=100):
        super(ResNet_SL, self).__init__()
        
        if encoder_params['is_cifar'] == True:
            num_classes = 10
            
        model_func, feat_dim = model_dict[arch_name]
        self.encoder = model_func(**encoder_params)
        
        self.fc = nn.Linear(feat_dim, num_classes) 
        self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.bias.data.zero_()
            
    def forward(self, x):
        return self.fc(self.encoder(x))