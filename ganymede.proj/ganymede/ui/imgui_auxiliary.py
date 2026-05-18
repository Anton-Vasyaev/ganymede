# python
import logging
from dataclasses import dataclass
# 3rd party
import imgui
import glfw
import time
import OpenGL.GL as gl
from imgui.integrations.glfw import GlfwRenderer

from threading import Thread
from abc import ABC, abstractmethod


__GLFW_INIT = False





def impl_glfw_init(
    window_name : str = 'minimal ImGui/GLFW3 example', 
    width       : int = 1280, 
    height      : int = 720
):
    global __GLFW_INIT
    if __GLFW_INIT:
        return
    
    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    # OS X supports only forward-compatible core profiles from 3.2
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(int(width), int(height), window_name, None, None)
    glfw.make_context_current(window)

    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    return window


@dataclass 
class WindowInfo:
    width : int
    
    height : int


class IImguiProcessor(ABC):
    @abstractmethod
    def process(self, window_info : WindowInfo) -> None:
        raise NotImplementedError()
    

class IMGUIWindow(object):
    __IMGUI_AUX_LOGGER  = logging.getLogger('ganymede.imgui_auxiliary')
    __IMGUI_WINDOW_FONT = "C:/Windows/Fonts/CONSOLA.TTF" 
    
    __processor : IImguiProcessor
    
    __stop_flag : bool
    
    __thread : Thread
    
    
    
    
    def __process(
        self
    ):
        self.backgroundColor = (0, 0, 0, 1)
        self.window = impl_glfw_init()
        gl.glClearColor(*self.backgroundColor)
        imgui.create_context()
        io = imgui.get_io()
    
        # 1. Получаем доступ к атласу шрифтов
        # 2. Получаем диапазон кириллических символов
        ranges = io.fonts.get_glyph_ranges_cyrillic()
        
        # 3. Загружаем шрифт (укажите путь к любому .ttf с поддержкой кириллицы)
        # Если путь пустой или файл не найден, ImGui выдаст ошибку
        try:
            font = io.fonts.add_font_from_file_ttf(
                IMGUIWindow.__IMGUI_WINDOW_FONT, 
                14.0, 
                None, 
                ranges
            )
        except:
            IMGUIWindow.__IMGUI_AUX_LOGGER.warning(
                f'couldnt load font:{IMGUIWindow.__IMGUI_WINDOW_FONT}.'
            )
        
        self.impl = GlfwRenderer(self.window)

        while not glfw.window_should_close(self.window) and not self.__stop_flag:
            glfw.poll_events()
            self.impl.process_inputs()
            imgui.new_frame()
            
            # Получаем информацию об окне
            framebuffer_w, framebuffer_h = glfw.get_framebuffer_size(self.window)
            win_info = WindowInfo(framebuffer_w, framebuffer_h)
            
            self.__processor.process(win_info)

            imgui.render()

            gl.glClearColor(*self.backgroundColor)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            self.impl.render(imgui.get_draw_data())
            glfw.swap_buffers(self.window)


        self.__stop_flag = True
        self.impl.shutdown()
        glfw.terminate()
    
    
    def __init__(
        self,
        processor : IImguiProcessor
    ):
        super().__init__()
        
        self.__processor    = processor
        self.__stop_flag    = False
        self.__thread       = Thread(target=self.__process, daemon=True)
        
        
    def __del__(self):
        self.__stop_flag = True
        self.join()
        

    def start(self):
        self.__thread.start()
        
        
    def stop(self):
        self.__stop_flag = True


    def join(self):
        self.__thread.join()
        
        
    def is_stopped(self) -> bool:
        return self.__stop_flag