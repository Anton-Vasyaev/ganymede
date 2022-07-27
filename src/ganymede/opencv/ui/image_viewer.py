# project
import ganymede.core    as g_core
import ganymede.imaging as g_img

from ganymede.draw import draw_text_list
from .funcs import imshow

class ImageViewer:
    def __init__(
        self, 
        image_provider
    ):
        self.image_provider = image_provider
        
        self.cursor_idx = 0


    def imshow(
        self,
        label           : str, 
        wait_ms         : int  = 0, 
        escape_catсh    : bool = True
    ):
        img = self.image_provider[self.cursor_idx] 
        images_len = len(self.image_provider)

        draw_text = [
            f'{self.cursor_idx + 1}/{images_len}',
            (0.05, 0.05),
            (255, 0, 0)
        ]

        if g_img.get_channels(img) == 1:
            img = g_img.cast_one_channel_img(img, 3)

        draw_text_list(
            img,
            [draw_text]
        )

        key = imshow(label, img, wait_ms, escape_catch=escape_catсh)
        if   key == 97:  self.cursor_idx = g_core.ring_add(self.cursor_idx, -1, images_len)
        elif key == 100: self.cursor_idx = g_core.ring_add(self.cursor_idx,  1, images_len)

        return key