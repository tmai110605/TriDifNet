import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import conv1x1_block, conv3x3_block, conv3x3_dw_blockAll, Classifier
from .TriDiff2 import TensorDiff3D


class TriDiffBlock2(nn.Module):
    """
    Core building block of TriDiffNet2.
    Identical to TriDiffBlock in TriDiffNet except that the full TriDiffModule
    (TASA + TensorDiff3D) is replaced by TensorDiff3D alone — an ablation that
    isolates the contribution of the 3D gradient-field operator without the
    strip-attention overhead.
    - Multi-scale / Multi-stride depthwise feature extraction (MuS)
    - Full-Residual (FR) identity mapping preserving gradient flow
    - 3D Tensor Differential Operator only (TASA removed)
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1):
        super(TriDiffBlock2, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 1. Pointwise projection (groups=2 for C >= 192 to bound parameter complexity)
        groups_pw1 = 2 if (in_channels >= 192 and in_channels % 2 == 0) else 1
        self.pw1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=in_channels,
            use_bn=False,
            activation=None,
            groups=groups_pw1
        )

        # 2. Multi-Stride Depthwise branch
        self.dw1 = conv3x3_dw_blockAll(channels=in_channels, stride=stride)
        if stride == 1:
            self.dw2 = conv3x3_dw_blockAll(channels=in_channels, stride=2)

        # 3. Output Pointwise convolution
        self.pw2 = conv1x1_block(in_channels=in_channels, out_channels=out_channels)

        # 4. 3D Tensor Differential Operator (replaces full TriDiffModule)
        self.diff = TensorDiff3D(out_channels)

        # 5. Full-Residual Connection
        if stride != 1 or in_channels != out_channels:
            self.pointwise_residual = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                activation=None
            )
        else:
            self.pointwise_residual = None

    def forward(self, x):
        residual = x

        # Pw1
        x = self.pw1(x)

        # Multi-stride depthwise convolution
        x1 = self.dw1(x)
        if self.stride == 1:
            batch_size, num_channels, h, w = x1.size()
            x2 = self.dw2(x)
            x2 = F.interpolate(x2, size=(h, w), mode='nearest')
            x = F.relu(x1 * torch.sigmoid(x2))
        else:
            x = x1

        # Pw2
        x = self.pw2(x)

        # 3D Tensor Differential enhancement (identity + gradient field)
        x = x + self.diff(x)

        # Full-Residual Connection
        if self.pointwise_residual is not None:
            residual = self.pointwise_residual(residual)
        x = x + residual
        return x


class TriDiffNet2(nn.Module):
    """
    TriDiffNet2: ablation variant of TriDiffNet with TASA removed.
    Retains the full MuS depthwise structure and Full-Residual connections
    of TriDiffNet; replaces TriDiffModule with TensorDiff3D only.
    """
    def __init__(self,
                 num_classes,
                 init_conv_channels,
                 init_conv_stride,
                 channels,
                 strides,
                 in_channels=3,
                 in_size=(224, 224),
                 use_data_batchnorm=True,
                 dropout_rate=0.2,
                 final_conv_channels=512):
        super(TriDiffNet2, self).__init__()
        self.in_size = in_size
        self.use_data_batchnorm = use_data_batchnorm
        self.final_conv_channels = final_conv_channels

        self.backbone = nn.Sequential()

        # Data batchnorm
        if self.use_data_batchnorm:
            self.backbone.add_module("data_bn", nn.BatchNorm2d(num_features=in_channels))

        # Initial stem convolution
        self.backbone.add_module(
            "init_conv",
            conv3x3_block(
                in_channels=in_channels,
                out_channels=init_conv_channels,
                stride=init_conv_stride
            )
        )

        # Stages construction
        current_channels = init_conv_channels
        for stage_id, stage_channels in enumerate(channels):
            stage = nn.Sequential()
            for unit_id, unit_channels in enumerate(stage_channels):
                stride = strides[stage_id] if unit_id == 0 else 1
                stage.add_module(
                    f"unit{unit_id + 1}",
                    TriDiffBlock2(
                        in_channels=current_channels,
                        out_channels=unit_channels,
                        stride=stride,
                    )
                )
                current_channels = unit_channels
            self.backbone.add_module(f"stage{stage_id + 1}", stage)

        # Final projection head
        self.backbone.add_module(
            "final_conv",
            conv1x1_block(
                in_channels=current_channels,
                out_channels=self.final_conv_channels,
                activation="relu"
            )
        )
        if dropout_rate > 0:
            self.backbone.add_module("dropout1", nn.Dropout2d(p=dropout_rate))
        self.backbone.add_module("global_pool", nn.AdaptiveAvgPool2d(output_size=1))
        if dropout_rate > 0:
            self.backbone.add_module("dropout2", nn.Dropout2d(p=dropout_rate))

        # Classifier
        self.classifier = Classifier(in_channels=self.final_conv_channels, num_classes=num_classes)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.classifier.init_params()

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


def build_TriDiffNet(num_classes,
                      typesize='compact',
                      cifar=False,
                      width_multiplier=1.0,
                      dropout_rate=0.2):
    """
    Factory function for TriDiffNet2:
    - 'compact': Harmonic Pyramidal Progression (~1.04M params, same schedule as TriDiffNet compact)
    - 'ultra':   Ultra-compact edge configuration (~720K params)
    """
    init_conv_channels = 32

    if typesize == 'compact':
        # Same channel schedule as TriDiffNet compact
        channels = [
            [48],
            [64, 96],
            [144, 192, 144],
            [256, 288],
            [384]
        ]
        final_conv_channels = 512
    elif typesize == 'ultra':
        # Same channel schedule as TriDiffNet ultra
        channels = [
            [32],
            [48, 64],
            [128, 160, 128],
            [192, 224],
            [320]
        ]
        final_conv_channels = 448
    else:
        raise ValueError(f"Unknown typesize: '{typesize}'. Supported: 'compact', 'ultra'.")

    if cifar:
        in_size = (32, 32)
        init_conv_stride = 1
        strides = [1, 1, 2, 2, 2]
    else:
        in_size = (224, 224)
        init_conv_stride = 2
        strides = [1, 2, 2, 2, 2]

    if width_multiplier != 1.0:
        channels = [[int(c * width_multiplier) for c in s] for s in channels]
        init_conv_channels = int(init_conv_channels * width_multiplier)
        final_conv_channels = int(final_conv_channels * width_multiplier)

    return TriDiffNet2(
        num_classes=num_classes,
        init_conv_channels=init_conv_channels,
        init_conv_stride=init_conv_stride,
        channels=channels,
        strides=strides,
        in_size=in_size,
        dropout_rate=dropout_rate,
        final_conv_channels=final_conv_channels
    )
