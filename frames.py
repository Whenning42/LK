import av
import os
import time
import matplotlib.pyplot as plt
import sdl2
import sdl2.ext
import numpy as np

WINDOW_SIZE = (1280, 720)

#initialize sdl2
sdl2.ext.init()
window = sdl2.ext.Window("Images", size = WINDOW_SIZE)
window.show()
window_surf = sdl2.SDL_GetWindowSurface(window.window)
window_array = sdl2.ext.pixels3d(window_surf.contents)


start_time = time.time()
avr_frame_time = 1/60.0

def downsample(im, block_x, block_y):
    assert(im.shape[0] % block_x == 0)
    assert(im.shape[1] % block_y == 0)

    n = block_x * block_y
    v = np.zeros((im.shape[0] // block_x, im.shape[1] // block_y, im.shape[2]))
    for y in range(block_y):
        for x in range(block_x):
            v += 1/n * im[x::block_x, y::block_y, :]

    for y in range(block_y):
        for x in range(block_x):
            im[x::block_x, y::block_y, :] = v
    return im

container = av.open('minecraft.mp4')
for packet in container.demux():
    for frame in packet.decode():
        if isinstance(frame, av.video.frame.VideoFrame):

            im = frame.to_rgb().to_ndarray()
            im = downsample(im, 16, 16)

            window_array[: , :, 2::-1] = im.swapaxes(0, 1)
            window.refresh()

            frame_time = time.time() - start_time
            avr_frame_time = avr_frame_time * .9 + frame_time * .1
            print("FPS: ", 1.0 / avr_frame_time)
            start_time = time.time()
