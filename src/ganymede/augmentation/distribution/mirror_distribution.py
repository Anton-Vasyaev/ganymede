# python
from dataclasses import dataclass
# project
import ganymede.random as g_random
from ..parameters.mirror_parameters import MirrorParameters


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


    def generate(
        self,
        random_instance = None
    ):
        rs = random_instance

        horizontal = g_random.get_random_bool(random_instance=rs) if self.horizontal else False
        vertical   = g_random.get_random_bool(random_instance=rs) if self.vertical   else False

        return MirrorParameters(horizontal, vertical)
