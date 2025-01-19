import pycuda.driver as cuda
import numpy as np

class MemoryManager:
    """Manages device memory allocation and transfers."""

    def __init__(self, dtype=np.float32):
        """Initialize the MemoryManager with a default data type."""
        self.dtype = np.dtype(dtype)  # Ensure dtype is a numpy dtype

    def allocate_device_array(self, shape, dtype=None):
        """Allocate memory on the GPU for an array."""
        if dtype is None:
            dtype = self.dtype  # Use default dtype if not provided
        dtype = np.dtype(dtype)  # Ensure dtype is a numpy dtype
        size = int(np.prod(shape) * dtype.itemsize)  # Calculate size of memory needed
        device_array = cuda.mem_alloc(size)
        print(f"Allocated {size} bytes on the device.")
        return device_array

    def copy_to_device(self, host_array, device_array):
        """Copy data from host (CPU) to device (GPU)."""
        cuda.memcpy_htod(device_array, host_array)
        print(f"Copied data of shape {host_array.shape} to device.")

    def copy_to_host(self, device_array, host_array):
        """Copy data from device (GPU) to host (CPU)."""
        cuda.memcpy_dtoh(host_array, device_array)
        print(f"Copied data from device to host with shape {host_array.shape}.")

    def allocate_array(self, shape, dtype=None):
        """Allocate memory and return a numpy array interface."""
        return self.allocate_device_array(shape, dtype)

    def deallocate(self, device_array):
        """Deallocate memory on the GPU."""
        device_array.free()
        print("Deallocated device memory.")

    def allocate_array_like(self, host_array):
        """Allocate memory on the device that is the same shape and dtype as the host array."""
        return self.allocate_device_array(host_array.shape, host_array.dtype)

