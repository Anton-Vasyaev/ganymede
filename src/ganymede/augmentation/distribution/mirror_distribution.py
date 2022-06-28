# python
from dataclasses import dataclass


@dataclass
class MirrorDistribution:
    horizontal : True
    vertical   : True


    @staticmethod
    def load_from_dict(data):
        return MirrorDistribution(
            data['horizontal'],
            data['vertical']
        )