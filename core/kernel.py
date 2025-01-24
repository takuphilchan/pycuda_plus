import pycuda.compiler as compiler
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import numpy as np

class KernelExecutor:
    def __init__(self, prebuilt_kernels=None):
        """
        Initialize KernelExecutor with optional prebuilt kernels.
        
        Args:
            prebuilt_kernels (dict, optional): Dictionary of precompiled kernels
        """
        self.compiled_kernels = prebuilt_kernels or {}

    def compile_kernel(self, kernel_code, kernel_name):
        """
        Compile a CUDA kernel from source code.
        
        Args:
            kernel_code (str): CUDA kernel source code
            kernel_name (str): Name of the kernel function to compile
        
        Returns:
            Compiled CUDA kernel function
        """
        module = compiler.SourceModule(kernel_code)
        kernel = module.get_function(kernel_name)
        self.compiled_kernels[kernel_name] = kernel
        return kernel

    def get_kernel(self, kernel_name):
        """
        Retrieve a kernel by name, either prebuilt or manually compiled.
        
        Args:
            kernel_name (str): Name of the kernel
        
        Returns:
            Compiled CUDA kernel function
        """
        if kernel_name not in self.compiled_kernels:
            raise ValueError(f"Kernel {kernel_name} not found")
        return self.compiled_kernels[kernel_name]

    def launch_kernel(self, kernel, grid, block, *args):
        """
        Launch a CUDA kernel with the specified configuration.
        
        Args:
            kernel (function): Compiled CUDA kernel function
            grid (tuple): Grid dimension configuration
            block (tuple): Block dimension configuration
            *args: Kernel arguments
        """
        kernel(*args, grid=grid, block=block)