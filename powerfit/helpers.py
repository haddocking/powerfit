import numpy as np
try:
    import pyopencl as cl
except ImportError:
    pass

def resolution2sigma(resolution):
    return resolution/(np.sqrt(2.0) * np.pi)

def sigma2resolution(sigma):
    return sigma * (np.sqrt(2.0) * np.pi)

def get_queue(platformid=0, deviceid=0):
    try:
        platform = cl.get_platforms()[platformid]
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        context = cl.Context(devices=devices)
        queue = cl.CommandQueue(context, device=devices[deviceid])
    except:
        queue = None

    return queue

