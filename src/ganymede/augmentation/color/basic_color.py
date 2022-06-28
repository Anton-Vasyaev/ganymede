# 3rd party
import numpy as np
# project
import ganymede.imaging as g_imaging


def augmentation_basic_color_img(
    img                      : np.ndarray,
    red                      : float,
    green                    : float,
    blue                     : float,
    normalize_floating       : bool = True,
    convert_to_original_type : bool = True
):
    dtype = img.dtype

    mod_value = None
    if normalize_floating and np.issubdtype(dtype, np.floating):
        mod_value = 1.0
    else:
        mod_value = 255

    img = img.astype(np.float32)
    channels = g_imaging.get_channels(img)

    if channels == 3:
        img[:,:,0] *= blue
        img[:,:,1] *= green
        img[:,:,2] *= red
    elif channels == 1:
        img *= 0.299 * red + 0.587 * green + 0.114 * blue

    img = img.clip(0, mod_value)

    if convert_to_original_type and not np.issubdtype(dtype, np.floating):
        img = img.astype(dtype)
        
    return img