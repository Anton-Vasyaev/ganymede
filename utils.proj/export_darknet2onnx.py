# python
import argparse
import os.path as path
from typing import List
# 3rd party
import torch
from pathlib import Path
# project
from ganymede.ml.pytorch.models.darknet import DarknetModel

def get_export_file_name(cfg_name : str) -> str:
    point_idx = cfg_name.rfind('.')
    if point_idx < 0:
        return f'{cfg_name}.onnx'
    else:
        return f'{cfg_name[:point_idx]}.onnx'
    

def export_darknet2onnx(args):
    cfg_path     : str = args.cfg_path
    weights_path : str = args.weights_path
    export_dir   : str = args.export_dir

    Path(export_dir).mkdir(parents=True, exist_ok=True)

    export_file_name = get_export_file_name(Path(cfg_path).stem)
    export_file_path = path.join(path.abspath(export_dir), export_file_name)

    model = DarknetModel.load_from_file(cfg_path)
    model.load_weights_from_file(weights_path)

    input_w = model.net_params.width
    input_h = model.net_params.height
    input_c = model.net_params.channels

    dummy_input = torch.randn(1, input_c, input_h, input_w)

    dynamic_axes = dict()

    input_names = ['input']
    dynamic_axes['input'] = { 0 : 'batch_size' }

    outputs_count = len(model.get_output_modules())
    outputs_names : List[str] = list()
    for output_idx in range(outputs_count):
        output_name = f'output_{output_idx}'
        outputs_names.append(output_name)
        dynamic_axes[output_name] = { 0 : 'batch_size' }

    torch.onnx.export(
        model,
        dummy_input,
        export_file_path,
        verbose=True,
        input_names=input_names,
        output_names=outputs_names,
        dynamic_axes=dynamic_axes
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, required=True, help='path to darknet config file \'*.cfg\'.')
    parser.add_argument('--weights_path', type=str, required=True, help='path to darknet weights file \'*.weights\'.')
    parser.add_argument('--export_dir', type=str, required=False, default='.', help='path to export directory.')

    args = parser.parse_args()

    export_darknet2onnx(args)