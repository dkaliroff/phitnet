import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def inception_branch(in_planes,inception_params):
    k_size=inception_params[2]
    batch_norm=inception_params[3]
    reflection_pad=inception_params[4]
    if batch_norm == True:
        return nn.Sequential(
            nn.Conv2d(in_planes, inception_params[0], kernel_size=1),
            nn.BatchNorm2d(inception_params[0]),
            nn.ELU(inplace=True),
            nn.Conv2d(inception_params[0], inception_params[1], kernel_size=k_size, padding=(k_size - 1) // 2),
            nn.BatchNorm2d(inception_params[1]),
            nn.ELU(inplace=True)
        )
    else:
        if reflection_pad:
            return nn.Sequential(
                nn.Conv2d(in_planes, inception_params[0], kernel_size=1),
                nn.ELU(inplace=True),
                nn.ReflectionPad2d((k_size - 1) // 2),
                nn.Conv2d(inception_params[0], inception_params[1], kernel_size=k_size, padding=0),
                nn.ELU(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, inception_params[0], kernel_size=1),
                nn.ELU(inplace=True),
                nn.Conv2d(inception_params[0], inception_params[1], kernel_size=k_size, padding=(k_size-1)//2),
                nn.ELU(inplace=True)
        )


class Interpolate(nn.Module):
    def __init__(self, scale_factor,mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x

def upsample(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=3,stride=2,padding=1)
    elif mode =='bilinear':
        # out_channels is always going to be the same
        # as in_channels
        upsample = Interpolate(scale_factor=2, mode='bilinear')
        return nn.Sequential(
            upsample,
        )


def downsample(in_channels, out_channels, mode):
    if mode == 'conv':
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1)
    elif mode =='avgpool':
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
    elif mode == 'maxpool':
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


def crop_like(input, ref):
    assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
    return input[:, :, :ref.size(2), :ref.size(3)]

def conv1x1(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1)

def group3x3(in_planes, out_planes, stride=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes*16, kernel_size=3, stride=stride, padding=1),
        nn.ELU(inplace=True),
        nn.Conv2d(out_planes*16, out_planes, kernel_size=3, groups=out_planes, padding=1),
    )

def conv3x3(in_planes, out_planes, stride=1, bias=True):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

def final_act_layer(act):
    if act == 'softmax':
        return nn.Hardtanh()
    if act == 'tanh':
        return nn.Tanh()
    if act == 'sigmoid':
        return nn.Sigmoid()
    if act == 'sigmoid4x':
        return Sigmoid4x()

class Sigmoid4x(nn.Module):
    def forward(self, input):
        return torch.sigmoid(4*input)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels,inception_params, downmode='maxpool', pooling=True):
        super(DownConv, self).__init__()
        self.inception_params=inception_params
        self.pooling = pooling
        self.in_channels=in_channels
        self.inception_layer = inception_branch(self.in_channels,self.inception_params)

        if self.pooling:
            self.pool = downsample(inception_params[1], inception_params[1], downmode)
            # self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.inception_layer(x)
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool

class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels,inception_params,upmode):
        super(UpConv, self).__init__()
        self.in_channels=in_channels
        self.inception_params=inception_params
        self.inception_layer = inception_branch(self.in_channels,self.inception_params)
        self.upsample=upsample(inception_params[1], inception_params[1],upmode)

    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upsample(from_up)
        if self.training is False:
            from_up_cropped = crop_like(from_up, from_down)
        else:
            from_up_cropped=from_up
        x = torch.cat((from_up_cropped, from_down), 1)
        x = self.inception_layer(x)
        return x

class InvRepNet(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597
    """

    def __init__(self, out_channels=3, in_channels=3, depth=5,
                 final_act=None, final_conv='3x3', upmode='bilinear',downmode='maxpool',
                 channel_wise=False,inception_params=[32,64,5,False,False],inference_ref_padded=False):
        """
        """
        super(InvRepNet, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.depth = depth
        self.final_act = final_act
        self.final_conv = final_conv
        self.upmode = upmode
        self.downmode = downmode
        self.down_convs = []
        self.up_convs = []
        self.channel_wise=channel_wise
        self.pad_len=2**depth
        self.inference_ref_padded=inference_ref_padded

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else inception_params[1]
            pooling = True if i < depth - 1 else False

            down_conv = DownConv(ins, inception_params, self.downmode, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth - 1):
            up_conv = UpConv(inception_params[1]*2,inception_params,self.upmode)
            self.up_convs.append(up_conv)

        if self.final_conv == '3x3':
            self.final_conv_layer = conv3x3(inception_params[1], self.out_channels)
        elif self.final_conv == 'group3x3':
            self.final_conv_layer = group3x3(inception_params[1]+self.in_channels, self.out_channels)
        else:
            self.final_conv_layer = conv1x1(inception_params[1], self.out_channels)

        if self.final_act is not None:
            self.final_act_layer = final_act_layer(self.final_act)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, input):
        if self.inference_ref_padded and not self.training:
            o_shape = input[0].shape
            input_ref_padded = [F.pad(input[0], (self.pad_len, self.pad_len, self.pad_len, self.pad_len), 'reflect')]
        else:
            input_ref_padded=input

        if not self.channel_wise:
            output = []
            for ii, x in enumerate(input_ref_padded):
                encoder_outs = []
                # encoder pathway, save outputs for merging
                for i, module in enumerate(self.down_convs):
                    x, before_pool = module(x)
                    encoder_outs.append(before_pool)

                for i, module in enumerate(self.up_convs):
                    before_pool = encoder_outs[-(i + 2)]
                    x = module(before_pool, x)

                #final conv o desired ammount of channels in output
                if self.final_conv=='group3x3':
                    x = self.final_conv_layer(torch.cat((x,input[ii]),1))
                else:
                    x = self.final_conv_layer(x)

                #final activation if no None
                if self.final_act is not None:
                    if self.final_act == 'clamp01':
                        x = torch.clamp(x, min=0.0, max=1.0)
                    else:
                        x= self.final_act_layer(x)
                output.append(x)

        else:
            output = []
            for ii, x_ch in enumerate(input_ref_padded):
                x_ch_out=torch.zeros_like(x_ch)
                for ch in range(3):
                    x=x_ch[:,ch,:,:].unsqueeze(1)
                    encoder_outs = []
                    # encoder pathway, save outputs for merging
                    for i, module in enumerate(self.down_convs):
                        x, before_pool = module(x)
                        encoder_outs.append(before_pool)

                    for i, module in enumerate(self.up_convs):
                        before_pool = encoder_outs[-(i + 2)]
                        x = module(before_pool, x)

                    # final conv o desired ammount of channels in output
                    if self.final_conv == 'group3x3':
                        x = self.final_conv_layer(torch.cat((x, input[ii]), 1))
                    else:
                        x = self.final_conv_layer(x)

                    # final activation if no None
                    if self.final_act is not None:
                        if self.final_act == 'clamp01':
                            x = torch.clamp(x, min=0.0, max=1.0)
                        else:
                            x = self.final_act(x)
                    x_ch_out[:,ch,:,:]=x.squeeze(1)
                output.append(x_ch_out)

        if self.inference_ref_padded and not self.training:
            output_final = [output[0][:, :, self.pad_len:self.pad_len + o_shape[-2], self.pad_len:self.pad_len + o_shape[-1]]]
        else:
            output_final=output
        return output_final





