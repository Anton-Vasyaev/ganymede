# python
from typing      import List
from dataclasses import dataclass
# 3rd party
import torch
from pathlib import Path
# project
import ganymede.json as g_json
# project
from .train_parameters       import TrainParameters
from .environment_parameters import EnvironmentParameters


@dataclass
class CheckpointInfo:
    path        : str
    idx         : int
    accuracy    : float


class TrainState:
    train_params : TrainParameters
    env_params   : EnvironmentParameters


    @staticmethod
    def load_from_dict(config : dict):
        train_params = TrainParameters.load_from_dict(config['train'])
        env_params   = EnvironmentParameters.load_from_dict(config['environment'])

        return TrainState(train_params, env_params)
        

    @staticmethod
    def load_from_config_file(path):
        config = g_json.load_from_file(path)

        return TrainState.load_from_dict(config)


    def __init__(
        self,
        train_params : TrainParameters,
        env_params   : EnvironmentParameters
    ):
        self.train_params = train_params
        self.env_params    = env_params

        Path(env_params.checkpoint_path).mkdir(parents = True, exist_ok = True)
        Path(env_params.export_path).mkdir(parents = True, exist_ok = True)




    def _get_checkpoint(self) -> List[CheckpointInfo]:
        checkpoint_dir_p = Path(self.env_params.checkpoint_path)

        checkpoints : List[CheckpointInfo] = []
        for file_p in checkpoint_dir_p.iterdir():
            if not file_p.suffix == '.pth': continue

            file_name = file_p.stem
            idx_str, accuracy_str = file_name.split('_')[-2:]
            idx      = int(idx_str)
            accuracy = float(accuracy_str)

            checkpoints.append(
                CheckpointInfo(str(file_p), idx, accuracy)
            )

        checkpoints = sorted(checkpoints, key = lambda x : x.idx)

        return checkpoints


    def restore_last_checkpoint(
        self, 
        model : torch.nn.Module
    ) -> CheckpointInfo:
        checkpoints = self._get_checkpoint()

        checkpoint = checkpoints[-1]

        model.load_state_dict(
            torch.load(
                checkpoint.path, 
                map_location=torch.device('cpu')
            )
        )

        return checkpoint


    def restore_checkpoint_idx(
        self, 
        model : torch.nn.Module, 
        idx   : int
    ) -> CheckpointInfo:
        checkpoints = self._get_checkpoint()
        checkpoints = [x for x in checkpoints if x.idx == idx]

        if len(checkpoints) == 0:
            raise Exception(f'invalid epoch idx for checkpoint:{idx}')

        checkpoint_info = checkpoints[0]

        model.load_state_dict(
            torch.load(
                checkpoint_info.path,
                map_location=torch.device('cpu')
            )
        )
        
        return checkpoint_info


    def load_model_from_argument_idx(self, model, arg_idx) -> CheckpointInfo:
        if arg_idx != -1:
            if arg_idx == 0:
                return self.restore_last_checkpoint(model)
            else:
                return self.restore_checkpoint_idx(model, arg_idx)

        return CheckpointInfo('', 0, -1000000.0)


    def save_checkpoint(self, model, epoch_idx, accuracy):
        checkpoint_path = self.env_params.checkpoint_path
        in_w, in_h = self.train_params.input_size

        accuracy_str = '{:.3f}'.format(accuracy)

        class_name = model.__class__.__name__

        torch.save(
            model.state_dict(), 
            f'{checkpoint_path}/{class_name}_{in_w}x{in_h}_{epoch_idx}_{accuracy_str}.pth'
        )