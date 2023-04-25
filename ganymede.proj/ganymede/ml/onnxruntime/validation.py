# python
from typing import Tuple
# 3rd party
import onnxruntime as ort
# project
from .exception import ValidationException


def number_of_dimensions_equal(
    node         : ort.NodeArg, 
    number       : int, 
    session_name : str = 'unknown'
):
    if len(node.shape) != number:
        raise ValidationException(
            f'number of dimensions of node \'{node.name}\' in session \'{session_name}\' '
            f'is not equal {number}, gotted:{len(node.shape)}.'
        )


def is_shape_equal(
    node           : ort.NodeArg,
    required_shape : Tuple[int, ...],
    session_name   : str = 'unknown'
):
    node_shape : Tuple[int, ...] = node.shape

    number_of_dimensions_equal(node, len(required_shape))

    for node_dim, req_dim in zip(node.shape, required_shape):
        if node_dim != req_dim:
            raise ValidationException(
                f'shape of node \'{node.name}\' in session \'{session_name}\' != required shape:'
                f'{node_shape} != {required_shape}.'
            )
        

def is_dimension_equal(
    node         : ort.NodeArg,
    dim_index    : int,
    value        : int,
    session_name : str = 'unknown'
):
    if len(node.shape) <= dim_index:
        raise ValidationException(
            f'number of dimensions of node\'{node}\' in session \'{session_name}\' '
            f'<= {dim_index} (dimension index):{len(node.shape)}.'
        )
    
    if node.shape[dim_index] != value:
        raise ValidationException(
            f'dimension index ({dim_index}) of node \'{node}\' in session \'{session_name}\' '
            f'!= {value}:{node.shape[dim_index]}.'
        )



def is_convolutional_shape_format(
    node         : ort.NodeArg, 
    session_name : str = 'unknown'
):
    if len(node.shape) != 4:
        raise ValidationException(
            f'dimensions of node \'{node.name}\' in session \'{session_name}\' ' 
            f'is not convolutional format:{node.shape} (Expected 4 dims).'
        )
    

def is_equal_image_shape(
    node         : ort.NodeArg, 
    width        : int, 
    height       : int, 
    channels     : int,
    session_name : str = 'unknown'
):
    is_convolutional_shape_format(node, session_name)

    batch_size, node_channels, node_height, node_width = node.shape

    if node_channels != channels or node_height != height or\
        node_width != width:
        raise ValidationException(
            f'invalid image size of dimensions in node \'{node.name}\' in session \'{session_name}\':'
            f'({node_width}x{node_height}x{node_channels}) (node) != {width}x{height}x{channels} (required), '
            f'format: (width x height x channels).'
        )
    

def number_of_inputs_equal(
    sessions     : ort.InferenceSession,
    number       : int,
    session_name : str = 'unknown'
):
    inputs = sessions.get_inputs()
    if len(inputs) != number:
        raise ValidationException(
            f'number of inputs in session \'{session_name}\' is not equal {number}:{len(inputs)}.'
        )


def number_of_outputs_equal(
    sessions     : ort.InferenceSession,
    number       : int,
    session_name : str = 'unknown'
):
    outputs = sessions.get_outputs()
    if len(outputs) != number:
        raise ValidationException(
            f'number of outputs in session \'{session_name}\' is not equal {number}:{len(outputs)}.'
        )