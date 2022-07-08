# 3rd party
import torch.nn as nn
# project
from .efficientnet_pytorch import EfficientNet

'''
BACKBONES = {
    'efficient_net_b0' : lambda channels, output_size : EfficientNet.from_name(
        'efficientnet-b0', in_channels=channels, num_classes=output_size
    ),
    'efficient_net_b3' : lambda channels, output_size : EfficientNet.from_name(
        'efficientnet-b3', in_channels=channels, num_classes=output_size
    )
}
'''

class BackboneProvider:
    def __init__(self, backbone_name):
        self.backbone_name = backbone_name


    def __call__(self, channels, output_size):
        return EfficientNet.from_name(
            self.backbone_name, in_channels=channels, num_classes=output_size
        )

BACKBONES = { }

for idx in range(7):
    BACKBONES[f'efficient_net_b{idx}'] = BackboneProvider(f'efficientnet-b{idx}')



class EfficentKeynetV1(nn.Module):
    def __init__(
        self,
        backbone_name    : str,
        in_chanells      : int = 3,
        key_points_count : int = 4
    ):
        super(EfficentKeynetV1, self).__init__()
        
        self.in_channels = in_chanells
        self.output_size = key_points_count * 2

        self.backbone = BACKBONES[backbone_name](in_chanells, self.output_size)


    def __call__(self, x):
        x = self.backbone(x)

        batch_size = x.shape[0] 

        return x.view(batch_size, self.output_size // 2, 2)



class EfficentKeynetV2(nn.Module):
    def __init__(
        self,
        backbone_name    : str,
        in_chanells      : int = 3,
        key_points_count : int = 4
    ):
        super(EfficentKeynetV2, self).__init__()

        self.in_channels = in_chanells
        self.output_size = key_points_count * 2

        self.backbone = BACKBONES[backbone_name](in_chanells)

        self.linear_block = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(1000, 100),
            nn.Sigmoid(),
            nn.Linear(100, self.output_size)
        )


    def __call__(self, x):
        x = self.backbone(x)
        x = self.linear_block(x)

        batch_size = x.shape[0] 

        return x.view(batch_size, self.output_size // 2, 2)