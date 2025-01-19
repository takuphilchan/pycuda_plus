import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import numpy as np
from pycuda_plus.core.kernel import KernelExecutor

class NumpyHelper:
    """Utilities for advanced integration of NumPy arrays with PyCUDA."""

    def __init__(self, device_id=0):
        """
        Initialize NumpyHelper instance and set up the CUDA context.

        Args:
            device_id (int): The ID of the CUDA device to use.
        """
        self.device_id = device_id
        cuda.init()
        self.device = cuda.Device(self.device_id)
        self.context = self.device.make_context()
        print(f"CUDA device {self.device.name()} selected.")

    def reshape_device_array(self, device_array, new_shape):
        """
        Reshape a device array into a new shape (without changing its contents).

        Args:
            device_array (pycuda.gpuarray.GPUArray): Input device array.
            new_shape (tuple): The new shape for the device array.

        Returns:
            pycuda.gpuarray.GPUArray: Reshaped device array.
        """
        if np.prod(device_array.shape) != np.prod(new_shape):
            raise ValueError("Total number of elements must remain the same during reshaping.")
        return device_array.reshape(new_shape)
        
    def generate_patterned_array(self, shape, pattern):
        """
        Generate a patterned device array (e.g., a range or linspace).

        Args:
            shape (tuple): Shape of the array.
            pattern (str): Pattern type ('range', 'linspace').

        Returns:
            pycuda.gpuarray.GPUArray: Patterned device array.
        """
        size = np.prod(shape)
        if pattern == 'range':
            host_array = np.arange(size, dtype=np.float32).reshape(shape)
        elif pattern == 'linspace':
            host_array = np.linspace(0, 1, size, dtype=np.float32).reshape(shape)
        else:
            raise ValueError(f"Unsupported pattern: {pattern}")
        return gpuarray.to_gpu(host_array)

    def batch_copy_to_device(self, numpy_arrays):
        """
        Copy multiple NumPy arrays to device memory in batch.

        Args:
            numpy_arrays (list of np.ndarray): List of NumPy arrays to copy.

        Returns:
            list of pycuda.gpuarray.GPUArray: List of corresponding device arrays.
        """
        return [gpuarray.to_gpu(arr) for arr in numpy_arrays]

    def batch_copy_to_host(self, device_arrays):
        """
        Copy multiple device arrays back to host memory in batch.

        Args:
            device_arrays (list of pycuda.gpuarray.GPUArray): List of device arrays.

        Returns:
            list of np.ndarray: List of corresponding NumPy arrays.
        """
        return [arr.get() for arr in device_arrays]

    def __del__(self):
        """Clean up the CUDA context when the helper object is deleted."""
        self.context.pop()
        print("CUDA context cleaned up.")
