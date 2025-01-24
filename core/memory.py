import pycuda.driver as cuda
import numpy as np
import cupy as cp

class MemoryManager:
    """Manages device memory allocation and transfers."""

    def __init__(self, dtype=np.float32):
        """Initialize the MemoryManager with a default data type."""
        self.dtype = np.dtype(dtype)

    def allocate_device_array(self, shape, dtype=None):
        """Allocate memory on the GPU for an array."""
        dtype = np.dtype(dtype if dtype is not None else self.dtype)
        size = int(np.prod(shape) * dtype.itemsize)
        device_array = cuda.mem_alloc(size)
        return device_array

    def copy_to_device(self, host_array, device_array):
        """Copy data from host (CPU) to device (GPU)."""
        cuda.memcpy_htod(device_array, host_array)

    def copy_to_host(self, device_array, host_array):
        """Copy data from device (GPU) to host (CPU)."""
        cuda.memcpy_dtoh(host_array, device_array)

    def deallocate(self, device_array):
        """Deallocate memory on the GPU."""
        device_array.free()

    def get_memory_info(self):
        """
        Retrieve the current GPU memory usage.
        
        Returns:
            tuple: (total_memory, used_memory), both in bytes.
        """
        free_memory, total_memory = cuda.mem_get_info()
        used_memory = total_memory - free_memory
        return total_memory, used_memory

    def numpy_to_device(self, np_array):
        """Allocate memory on the device and copy data from NumPy array to GPU."""
        device_array = self.allocate_device_array(np_array.shape, np_array.dtype)
        self.copy_to_device(np_array, device_array)
        return device_array

    def device_to_numpy(self, device_array, shape, dtype):
        """Copy data from GPU to NumPy array."""
        host_array = np.empty(shape, dtype=dtype)
        self.copy_to_host(device_array, host_array)
        return host_array

    def cupy_to_device(self, cp_array):
        """Copy data from a CuPy array to a PyCUDA device buffer."""
        device_array = self.allocate_device_array(cp_array.shape, cp_array.dtype)
        host_array = cp_array.get()
        self.copy_to_device(host_array, device_array)
        return device_array

    def device_to_cupy(self, device_array, shape, dtype):
        """Wrap a PyCUDA device buffer into a CuPy array."""
        size = int(np.prod(shape) * np.dtype(dtype).itemsize)
        memptr = cp.cuda.MemoryPointer(
            cp.cuda.UnownedMemory(int(device_array), size=size, owner=None), 0
        )
        return cp.ndarray(shape, dtype=dtype, memptr=memptr)

    def numpy_to_cupy(self, np_array):
        """Convert NumPy array to CuPy array."""
        return cp.asarray(np_array)

    def cupy_to_numpy(self, cp_array):
        """Convert CuPy array to NumPy array."""
        return cp_array.get()