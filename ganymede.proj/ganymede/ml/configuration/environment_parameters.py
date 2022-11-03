# python
from dataclasses import dataclass

@dataclass
class EnvironmentParameters:
    checkpoint_path : str
    export_path     : str

    @staticmethod   
    def load_from_dict(data):
        return EnvironmentParameters(
            data['checkpoint_path'],
            data['export_path']
        )