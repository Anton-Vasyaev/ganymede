import numpy as np

def default_input_processor(
    img_batch : np.ndarray
) -> None:
    img_batch /= 255.0
