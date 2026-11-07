import torch
import torch.nn

###
#%% Activation functions
###

class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class HSwish(torch.nn.Module):
    def forward(self, x):
        return x * torch.nn.functional.relu6(x + 3.0, inplace=True) / 6.0

class HSigmoid(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.relu6(x + 3.0, inplace=True) / 6.0

def get_activation(activation):
    if activation == "relu":
        return torch.nn.ReLU(inplace=True)
    elif activation == "relu6":
        return torch.nn.ReLU6(inplace=True)
    elif activation == "swish":
        return Swish()
    elif activation == "hswish":
        return HSwish()
    elif activation == "sigmoid":
        return torch.nn.Sigmoid()
    elif activation == "hsigmoid":
        return HSigmoid()
    else:
        raise NotImplementedError("Activation {} not implemented".format(activation))

###
#%% Misc modules
###

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Classifier(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_classes,
            kernel_size=1,
            bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return x

    def init_params(self):
        torch.nn.init.xavier_normal_(self.conv.weight, gain=1.0)
        if self.conv.bias is not None:
            torch.nn.init.constant_(self.conv.bias, 0)

###
#%% Conv layer wrappers
###

def conv1x1(in_channels, out_channels, stride=1, bias=False, groups=1):
    return torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=0,
        groups=groups,
        bias=bias)

def conv3x3(in_channels, out_channels, stride=1, bias=False, groups=1):
    return torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=groups,
        bias=bias)

def conv3x3_dw(channels, stride=1):
    return torch.nn.Conv2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=channels,
        bias=False)

###
#%% Conv block wrappers
###

class ConvBlock(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 bias=False,
                 use_bn=True,
                 activation="relu"):
        super().__init__()
        self.use_bn = use_bn
        self.use_activation = (activation is not None)

        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias)
        if self.use_bn:
            self.bn = torch.nn.BatchNorm2d(num_features=out_channels)
        if self.use_activation:
            self.activation = get_activation(activation)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_activation:
            x = self.activation(x)
        return x

def conv1x1_block(in_channels,
                  out_channels,
                  stride=1,
                  bias=False,
                  use_bn=True,
                  groups=1,
                  activation="relu"):
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=0,
        groups=groups,
        bias=bias,
        use_bn=use_bn,
        activation=activation)

def conv1x1_group_block(in_channels,
                        out_channels,
                        stride=1,
                        bias=False,
                        use_bn=True,
                        groups=2,
                        activation="relu"):
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=0,
        groups=groups,
        bias=bias,
        use_bn=use_bn,
        activation=activation)

def conv3x3_block(in_channels,
                  out_channels,
                  stride=1,
                  bias=False,
                  use_bn=True,
                  groups=1,
                  activation="relu"):
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=groups,
        bias=bias,
        use_bn=use_bn,
        activation=activation)

def conv3x3_dw_block(channels,
                     stride=1,
                     use_bn=True,
                     activation="relu"):
    return ConvBlock(
        in_channels=channels,
        out_channels=channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=channels,
        use_bn=use_bn,
        activation=activation)

def conv3x3_dw_blockAll(channels,
                        stride=1,
                        use_bn=True,
                        activation="relu",
                        padding=1):
    return ConvBlock(
        in_channels=channels,
        out_channels=channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        groups=channels,
        use_bn=use_bn,
        activation=activation)

