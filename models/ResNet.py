import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import math


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
            padding=0,
            dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 =  nn.Conv2d(out_channels, out_channels,\
                        kernel_size=kernel_size, stride=stride,\
                        padding=padding,\
                                dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
         
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out

class DeconvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
            padding=0,
            output_padding=0, dilation=1, upsample=None):
        super(DeconvBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 =  nn.ConvTranspose2d(out_channels, out_channels,\
                        kernel_size=kernel_size, stride=stride,\
                        padding=padding, output_padding=output_padding,\
                                dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

        self.upsample = upsample

    def forward(self, x):
        
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
         
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        #if self.upsample is not None:
        #    identity = self.upsample(x)

        if self.upsample is not None:
             identity = self.upsample(x)
        identity = F.interpolate(identity, size=(out.shape[-2],
            out.shape[-1]), mode="bilinear", align_corners=True)

        out = out + identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, downblock, upblock, layer_infos, n_classes, inp=None):
        super(ResNet, self).__init__()

        # first layer's channel #
        # conv-bn-relu-0
        in_channels, kernel_size, stride, padding\
                = tuple(layer_infos[0])
        self.in_channels = in_channels
        if inp is not None:
            self.conv1 = nn.Conv2d(2*n_classes, self.in_channels,
                    kernel_size=kernel_size,
                    stride=stride, padding=padding, bias=False)
        else:
            self.conv1 = nn.Conv2d(n_classes, self.in_channels,
                    kernel_size=kernel_size,
                    stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.LeakyReLU(inplace=True)

        # maxpool
        kernel_size, stride, padding = tuple(layer_infos[1])
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)


        # downlayers
        self.dlayer1 = self._make_downlayer(downblock, layer_infos[2])
        self.dlayer2 = self._make_downlayer(downblock, layer_infos[3])
        self.dlayer3 = self._make_downlayer(downblock, layer_infos[4])
        self.dlayer4 = self._make_downlayer(downblock, layer_infos[5])
        
        # uplayers
        self.uplayer1 = self._make_uplayer(upblock, layer_infos[6])
        self.uplayer2 = self._make_uplayer(upblock, layer_infos[7])
        self.uplayer3 = self._make_uplayer(upblock, layer_infos[8])
        self.uplayer4 = self._make_uplayer(upblock, layer_infos[9])
        self.uplayer5 = self._make_uplayer(upblock, layer_infos[10])


        upsample = None

        self.uplayer_top = upblock(self.in_channels, self.in_channels,
                kernel_size=3, stride=2, padding=1, output_padding=1, upsample=upsample)
        if inp is not None:

            self.conv1_1 = nn.Conv2d(self.in_channels, inp, kernel_size=1,
                    stride=1, bias=False)
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = None
            self.conv1_1 = nn.Conv2d(self.in_channels, n_classes, kernel_size=1, stride=1, bias=False)
    def _make_downlayer(self, block, layer_info):
        num_layer, out_channels, kernel_size, stride, padding, dilation\
                = tuple(layer_info)
        

        downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
        )
        layers = []
        layers.append(block(self.in_channels, out_channels, kernel_size,
            stride, padding, dilation, downsample))
        self.in_channels = out_channels
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, self.in_channels))
        return nn.Sequential(*layers)
    def _make_uplayer(self, block, layer_info):
        num_layer, out_channels, kernel_size, stride, padding, output_padding, dilation\
                = tuple(layer_info)
        

        if self.in_channels != out_channels:
            upsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            upsample = None
        layers = []
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, self.in_channels))
        layers.append(block(self.in_channels, out_channels, kernel_size,
            stride, padding, output_padding, dilation, upsample))

        self.in_channels = out_channels

        return nn.Sequential(*layers)
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.dlayer1(x)
        x = self.dlayer2(x)
        x = self.dlayer3(x)
        x = self.dlayer4(x)
     #   print(x.shape)


        x = self.uplayer1(x)
        x = self.uplayer2(x)
        x = self.uplayer3(x)
        x = self.uplayer4(x)
        x = self.uplayer5(x)

        x = self.uplayer_top(x)

        x = self.conv1_1(x)
        if self.sigmoid is not None:
            x = self.sigmoid(x)
        return x

if __name__ == "__main__":
    
    device = torch.device('cuda')
    
    '''
    # stride only
    layer_infos = [
            [64, 7, 2, 3],#first conv
            [3, 2, 1],#max pooling
            [3, 64, 3, 2, 1, 1],
            [4, 128, 3, 2, 1, 1],
            [6, 256, 3, 2, 1, 1],
            [3, 512, 3, 2, 1, 1],
            [3, 512, 3, 2, 1, 1, 1],
            [6, 256, 3, 2, 1, 1, 1],
            [4, 128, 3, 2, 1, 1, 1],
            [3, 64, 3, 2, 1, 1, 1],
            [1, 64, 3, 2, 1, 1, 1]
            ]
    '''
    '''
    # stride only, deeper
    layer_infos = [
            [64, 7, 2, 3],
            [3, 2, 1],
            [6, 64, 3, 2, 1, 1],
            [8, 128, 3, 2, 1, 1],
            [12, 256, 3, 2, 1 ,1],
            [6, 512, 3, 2 ,1 ,1],
            [6, 512, 3, 2, 1, 1, 1],
            [12, 256, 3, 2, 1, 1, 1],
            [8, 128, 3, 2, 1, 1, 1],
            [6, 64, 3, 2, 1, 1, 1],
            [2, 64, 3, 2, 1, 1, 1]

            ]
    '''
    ''' 
    # stride + dilation
    layer_infos = [
            [64, 7, 2, 3],
            [3, 2, 1],
            [3, 64, 3, 2, 1, 1],
            [4, 128, 3, 2, 2, 2],
            [6, 256, 3, 2, 2, 2],
            [3, 512, 3, 2, 2, 2],
            [3, 512, 3, 2, 2, 1, 2],
            [6, 256, 3, 2, 2, 1, 2],
            [4, 128, 3, 2, 2, 1, 2],
            [3, 64, 3, 2, 1, 1, 1],
            [1, 64, 3, 2, 1, 1, 1]
            ]
'
    # stride + bigger kernel
    layer_infos = [
            [64, 7, 2, 3],
            [3, 2, 1],
            [3, 64, 5, 2, 2, 1],
            [4, 128, 5, 2, 2, 1],
            [6, 256, 5, 2, 2, 1],
            [3, 512, 5, 2, 2, 1],
            [3, 512, 5, 2, 2, 1, 1],
            [6, 256, 5, 2, 2, 1, 1],
            [4, 128, 5, 2, 2, 1, 1],
            [3, 64, 3, 2, 1, 1, 1],
            [1, 64, 3, 2, 1, 1, 1]
            ]
'''
    layer_infos = [
            
                  [64, 7, 2, 3],
                  [3, 2, 1],
                  [12, 64, 3, 2, 1, 1],
                  [16, 128, 3, 1, 1, 1],
                  [24, 256, 3, 1, 1, 1],
                  [12, 512, 3, 1, 1, 1],
                  [12, 512, 3, 1, 1, 0, 1],
                  [24, 256, 3, 1, 1, 0, 1],
                  [16, 128, 3, 1, 1, 0, 1],
                  [12, 64, 3, 2, 1, 1, 1],
                  [4, 64, 3, 2, 1, 1, 1]
            ]
    model = ResNet(Bottleneck, DeconvBottleneck, layer_infos, 41).to(device)
    x = torch.rand(4, 41, 128, 128).to(device)
    gt = torch.rand(4, 128, 128) * 41
    gt = gt.long().to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    while True:
        optimizer.zero_grad()
        output = model(x)
        print(output.shape)
        assert False, "Pause"
        loss = criterion(output, gt)
        loss.backward()
        optimizer.step()
