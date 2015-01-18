import numpy as np
import pyopencl as cl


def get_queue():
    platform = cl.get_platforms()[0]
    devices = platform.get_devices(device_type=cl.device_type.GPU)
    context = cl.Context(devices=devices)
    queue = cl.CommandQueue(context, device=devices[0])
    return queue
