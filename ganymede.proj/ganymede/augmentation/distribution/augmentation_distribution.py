# python
from dataclasses import dataclass
# project
from .basic_color_distribution  import BasicColorDistribution
from .mirror_distribution       import MirrorDistribution
from .padding_distribution      import PaddingDistribution
from .rotate2d_distribution     import Rotate2dDistribution
from .rotate3d_distribution     import Rotate3dDistribution
from .stretch_distribution      import StretchDistribution


@dataclass
class AugmentationDistribution:
    basic_color : BasicColorDistribution    = None
    mirror      : MirrorDistribution        = None
    padding     : PaddingDistribution       = None
    rotate2d    : Rotate2dDistribution      = None
    rotate3d    : Rotate3dDistribution      = None
    stretch     : StretchDistribution       = None


    @staticmethod
    def load_from_dict(data):
        basic_color = BasicColorDistribution.load_from_dict(data['basic_color']) if 'basic_color' in data else None
        mirror      = MirrorDistribution.load_from_dict(data['mirror']) if 'mirror' in data else None
        padding     = PaddingDistribution.load_from_dict(data['padding']) if 'padding' in data else None
        rotate2d    = Rotate2dDistribution.load_from_dict(data['rotate2d']) if 'rotate2d' in data else None
        rotate3d    = Rotate3dDistribution.load_from_dict(data['rotate3d']) if 'rotate3d' in data else None
        stretch     = StretchDistribution.load_from_dict(data['stretch']) if 'stretch' in data else None


        return AugmentationDistribution(
            basic_color,
            mirror,
            padding,
            rotate2d,
            rotate3d,
            stretch
        )