import pycuda.driver as cuda
import numpy as np
import cupy as cp
import pycuda.autoinit
import time

class MemoryManager:
    """Manages device memory allocation and transfers."""

    def __init__(self, dtype=np.float32):
        self.dtype = np.dtype(dtype)
        self.device_metadata = {}

    def allocate_device_array(self, shape, dtype=None):
        dtype = np.dtype(dtype if dtype else self.dtype)
        size = np.prod(shape) * dtype.itemsize
        device_array = cuda.mem_alloc(int(size))
        self.device_metadata[device_array] = {'shape': shape, 'dtype': dtype}
        return device_array

    def copy_to_device(self, host_array, device_array):
        cuda.memcpy_htod(device_array, host_array)

    def copy_to_host(self, device_array, host_array):
        cuda.memcpy_dtoh(host_array, device_array)

    def deallocate(self, *device_arrays):
        for device_array in device_arrays:
            device_array.free()
            self.device_metadata.pop(device_array, None)

    def get_memory_info(self):
        free_mem, total_mem = cuda.mem_get_info()
        return total_mem, total_mem - free_mem, free_mem

    def numpy_to_device(self, np_array):
        device_array = self.allocate_device_array(np_array.shape, np_array.dtype)
        self.copy_to_device(np_array, device_array)
        return device_array

    def device_to_numpy(self, device_array):
        metadata = self.device_metadata.get(device_array)
        if not metadata:
            raise ValueError("Device array not found in metadata.")
        host_array = np.empty(metadata['shape'], dtype=metadata['dtype'])
        self.copy_to_host(device_array, host_array)
        return host_array

    def cupy_to_device(self, cp_array):
        return self.numpy_to_device(cp_array.get())

    def device_to_cupy(self, device_array):
        metadata = self.device_metadata.get(device_array)
        if not metadata:
            raise ValueError("Device array not found in metadata.")
        return cp.asarray(self.device_to_numpy(device_array))

    def get_shape(self, device_array):
        """Get the shape of a device array."""
        if device_array in self.device_metadata:
            return self.device_metadata[device_array]['shape']
        else:
            raise ValueError("Device array not found in metadata.")

    def get_dtype(self, device_array):
        """Get the dtype of a device array."""
        if device_array in self.device_metadata:
            return self.device_metadata[device_array]['dtype']
        else:
            raise ValueError("Device array not found in metadata.")
