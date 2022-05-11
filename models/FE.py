import torch.nn.functional as F
import torch
import torch.nn as nn
import gc

class feature_extraction_conv(nn.Module):
    """
    UNet for HITNet
    """

    def __init__(self, args):
        # print('init')
        super(feature_extraction_conv, self).__init__()
        gc.collect()
        torch.cuda.empty_cache()
        
        self.conv1x_0 = nn.Sequential(
            BasicConv2d(3, 16, 3, 1, 1, 1),
            BasicConv2d(16, 16, 3, 1, 1, 1),
        )
        gc.collect()
        torch.cuda.empty_cache()
        self.conv2x_0 = nn.Sequential(
            BasicConv2d(16, 16, 2, 2, 0, 1),
            BasicConv2d(16, 16, 3, 1, 1, 1),
            BasicConv2d(16, 16, 3, 1, 1, 1)
        )
        gc.collect()
        torch.cuda.empty_cache()
        self.conv4x_0 = nn.Sequential(
            BasicConv2d(16, 24, 2, 2, 0, 1),
            BasicConv2d(24, 24, 3, 1, 1, 1),
            BasicConv2d(24, 24, 3, 1, 1, 1)
        )
        gc.collect()
        torch.cuda.empty_cache()
        self.conv8x_0 = nn.Sequential(
            BasicConv2d(24, 24, 2, 2, 0, 1),
            BasicConv2d(24, 24, 3, 1, 1, 1),
            BasicConv2d(24, 24, 3, 1, 1, 1)
        )
        gc.collect()
        torch.cuda.empty_cache()
        self.conv16x_0 = nn.Sequential(
            BasicConv2d(24, 32, 2, 2, 0, 1),
            BasicConv2d(32, 32, 3, 1, 1, 1),
            BasicConv2d(32, 32, 3, 1, 1, 1)
        )
        gc.collect()
        torch.cuda.empty_cache()
        
        self.conv16_8x_0 = unetUp(32, 24, 24)
        self.conv8_4x_0 = unetUp(24, 24, 24)
        self.conv4_2x_0 = unetUp(24, 16, 16)
        self.conv2_1x_0 = unetUp(16, 16, 16)
        gc.collect()
        torch.cuda.empty_cache()
        
        self.last_conv_1x = nn.Conv2d(16, 16, 1, 1, 0, 1, bias=False)
        self.last_conv_2x = nn.Conv2d(16, 16, 1, 1, 0, 1, bias=False)
        self.last_conv_4x = nn.Conv2d(24, 24, 1, 1, 0, 1, bias=False)
        self.last_conv_8x = nn.Conv2d(24, 24, 1, 1, 0, 1, bias=False)
        self.last_conv_16x = nn.Conv2d(32, 32, 1, 1, 0, 1, bias=False)

    def forward(self, x):
        gc.collect()
        torch.cuda.empty_cache()
        # print('feature_extraction_conv: forward start ')
        layer1x_0 = self.conv1x_0(x)
        layer2x_0 = self.conv2x_0(layer1x_0)
        layer4x_0 = self.conv4x_0(layer2x_0)
        layer8x_0 = self.conv8x_0(layer4x_0)
        layer16x_0 = self.conv16x_0(layer8x_0)
        
        gc.collect()
        torch.cuda.empty_cache()
        # print(layer1x_0.size()) # torch.Size([1, 16, 1100, 800])
        # print(layer2x_0.size()) # torch.Size([1, 16, 550, 400])
        # print(layer4x_0.size()) # torch.Size([1, 24, 275, 200])

        # print(layer8x_0.size())
        # test:torch.Size([1, 24, 137, 100]
        # kitti:torch.Size([1, 24, 48, 160])
        
        # print(layer16x_0.size())
        # test: torch.Size([1, 32, 68, 50])
        # kitti: torch.Size([1, 32, 24, 80])
        gc.collect()
        torch.cuda.empty_cache()
        
        layer8x_1 = self.conv16_8x_0(layer16x_0, layer8x_0)
        layer4x_1 = self.conv8_4x_0(layer8x_1, layer4x_0)
        layer2x_1 = self.conv4_2x_0(layer4x_1, layer2x_0)
        layer1x_1 = self.conv2_1x_0(layer2x_1, layer1x_0)
        gc.collect()
        torch.cuda.empty_cache()
        layer16x_1 = self.last_conv_16x(layer16x_0)
        layer8x_2 = self.last_conv_8x(layer8x_1)
        layer4x_2 = self.last_conv_4x(layer4x_1)
        layer2x_2 = self.last_conv_2x(layer2x_1)
        layer1x_2 = self.last_conv_1x(layer1x_1)
        
        gc.collect()
        torch.cuda.empty_cache()

        return [layer16x_1, layer8x_2, layer4x_2, layer2x_2, layer1x_2]  # 1/16, 1/8, 1/4, 1/2, 1/1


def BasicConv2d(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                  padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True, negative_slope=0.2),
    )


def BasicTransposeConv2d(in_channels, out_channels, kernel_size, stride, pad, dilation):
    gc.collect()
    torch.cuda.empty_cache()
    
    output_pad = stride + 2 * pad - kernel_size * dilation + dilation - 1
    gc.collect()
    torch.cuda.empty_cache()
    
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, pad, output_pad, dilation, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True, negative_slope=0.2),
    )


class unetUp(nn.Module):
    def __init__(self, in_c1, in_c2, out_c):
        super(unetUp, self).__init__()
        gc.collect()
        torch.cuda.empty_cache()
                
        self.up_conv1 = BasicTransposeConv2d(in_c1, in_c1//2, 2, 2, 0, 1)
        gc.collect()
        torch.cuda.empty_cache()
        
        self.reduce_conv2 = BasicConv2d(in_c1//2+in_c2, out_c, 1, 1, 0, 1)
        gc.collect()
        torch.cuda.empty_cache()
        
        self.conv = nn.Sequential(
            BasicConv2d(out_c, out_c, 3, 1, 1, 1),
        )
        gc.collect()
        torch.cuda.empty_cache()

    def forward(self, inputs1, inputs2):  # small scale, large scale
        # print('unetUp: forward start')
        # print(inputs1.size())
        # print(inputs2.size())
        
        layer1 = self.up_conv1(inputs1)
        # print(layer1.size())
        
        
# test
# torch.Size([1, 32, 68, 50])
# torch.Size([1, 24, 137, 100])
# torch.Size([1, 16, 136, 100])
## 137이랑 136 안맞아서 에러 -> 패딩시작 사이즈 맞춰서 해결

# kitti
# torch.Size([1, 32, 24, 80])
# torch.Size([1, 24, 48, 160])
# torch.Size([1, 16, 48, 160])
        
        # print(torch.cat([layer1, inputs2], 1))
        layer2 = self.reduce_conv2(torch.cat([layer1, inputs2], 1))
        output = self.conv(layer2)
        return output
