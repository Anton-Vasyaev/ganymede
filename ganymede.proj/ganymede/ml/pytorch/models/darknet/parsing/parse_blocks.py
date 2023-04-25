# python
import os.path as path
import re
from typing import List, Any, Callable, Optional
from dataclasses import dataclass
# project
from .data import *
from .error import DarknetConfigLoadException

SPACE_LINE_PATTERN = r'\s*'

BLOCK_TYPE_PATTERN = r'\s*\[\w+\]\s*'

OPTION_PATTERN = r'\s*\w+\s*=\s*.+\s*'

SPACE_SYMBOLS = set([' ', '\t', '\n'])

START_RAISE_MSG = 'Failed to parse YOLO darknet config'

DARKNET_PARSERS : Dict[DarknetLayerType, Callable[[ConfigBlock], Any]] = {
    DarknetLayerType.CONVOLUTIONAL: parse_convolutional,
    DarknetLayerType.MAXPOOL: parse_maxpool,
    DarknetLayerType.NETWORK: parse_net,
    DarknetLayerType.ROUTE: parse_route,
    DarknetLayerType.UPSAMPLE: parse_upsample,
    DarknetLayerType.YOLO: parse_yolo
}


@dataclass
class DarknetBackbone:
    net_params : NetParams

    layers : List[Any]

    def validate(self):
        # validate same classes for each output layer
        classes = -1

        find_output = False

        for layer in self.layers:
            if isinstance(layer, YoloLayer):
                find_output = True

                yolo_layer = cast(YoloLayer, layer)
                
                current_classes = yolo_layer.classes

                if classes == -1:
                    classes = current_classes
                else:
                    if current_classes != classes:
                        raise Exception(f'Different classes in output layers in darknet backbone.')
                    
        if not find_output:
            raise Exception(
                f'Not find output layer in darknet backbone'
            )
                    

    def get_classes(self) -> int:
        classes = -1

        for layer in self.layers:
            if isinstance(layer, YoloLayer):
                yolo_layer = cast(YoloLayer, layer)
                
                return yolo_layer.classes
            
        return classes


def is_space_line(data : str) -> bool:
    '''
    Checks the string contain only space symbols.

    Args:
        data (str): String.

    Returns:
        bool: Check status.
    '''

    for symbol in data:
        if not symbol in SPACE_SYMBOLS:
            return False
        
    return True


def has_space_split_symbol(data : str) -> bool:
    for symbol in data:
        if symbol is SPACE_SYMBOLS:
            return True
        
    return False


def parse_blocks(
    lines    : List[str], 
    cfg_path : Optional[str] = None
) -> List[ConfigBlock]:
    '''
    Parse blocks in darknet configuration.

    Args:
        lines (List[str]): Readed lines of darknet configuration

    Raises:
        ValueError: If lines contain syntax errors.

    Returns:
        List[dict]: collection of configuration blocks. Type of block stored in field 'type' 
    '''
    
    blocks : List[ConfigBlock] = []
    
    current_block : Optional[ConfigBlock] = None
    
    for line_idx in range(len(lines)):
        full_line = lines[line_idx]
        if len(full_line) == 0:
            continue
        
        line = full_line
        if line[-1] == '\n':
            line = line[:-1]
        
        comment_prefix_idx = line.find('#')
        if comment_prefix_idx != -1:
            line = line[:comment_prefix_idx]

        if re.fullmatch(SPACE_LINE_PATTERN, line):
            continue
            
        elif re.fullmatch(BLOCK_TYPE_PATTERN, line):
            open_brace_idx = line.find('[')
            close_brace_idx = line.find(']')
            
            block_type = line[open_brace_idx+1:close_brace_idx]
            
            if not current_block is None:
                blocks.append(current_block)
            
            current_block = ConfigBlock(block_type, dict(), line_idx + 1)

            
        elif re.fullmatch(OPTION_PATTERN, line):
            left, right = line.split('=')
            left  = left.strip()
            right = right.strip()
            
            if current_block is None:
                raise DarknetConfigLoadException(
                    f'Option line ({line_idx}) before block  declaration \'[block]\'', 
                    cfg_path, 
                    line_idx + 1
                )

            param = ConfigParameter(left, right, line_idx + 1)

            current_block.params[left] = param
        
        else:
            raise DarknetConfigLoadException(
                f'{START_RAISE_MSG}. Not match pattern for \'{full_line}\'',
                cfg_path,
                line_idx + 1
            )

    if not current_block is None:
        blocks.append(current_block)
        
    return blocks


def build_bone(config_info : ConfigInfo) -> DarknetBackbone:
    layers : List[Any] = []

    net_params : Optional[NetParams] = None

    for block in config_info.blocks:
        if not DarknetLayerType.contain_str_present(block.name):
            raise DarknetConfigLoadException(
                f'Not supported config block:{block.name}',
                config_info.path,
                block.line_number
            )

        layer_type = DarknetLayerType.from_str(block.name)

        params = DARKNET_PARSERS[layer_type](block)

        if layer_type == DarknetLayerType.NETWORK:
            net_params = cast(NetParams, params)
        else:
            layers.append(params)

    if net_params is None:
        raise DarknetConfigLoadException(
            f'Cannot find [net] or [network] block in config', 
            config_info.path
        )

    return DarknetBackbone(
        net_params,
        layers
    )
        

def read_darknet_bone_from_file(cfg_path : str) -> DarknetBackbone:
    if not path.exists(cfg_path):
        raise Exception(f'darknet config path is not exist:{cfg_path}.')

    with open(cfg_path, 'r') as fh:
        lines = fh.readlines()
        blocks = parse_blocks(lines, cfg_path)

        config_info = ConfigInfo(blocks, cfg_path)

        return build_bone(config_info)


def read_darknet_bone_from_str(data : str) -> DarknetBackbone:
    lines = data.split('\n')

    blocks = parse_blocks(lines, None)

    config_info = ConfigInfo(blocks, None)

    return build_bone(config_info)