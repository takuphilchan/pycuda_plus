import pycuda.compiler as compiler
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

class KernelExecutor:
    """Handles kernel compilation and execution for CUDA."""

    def __init__(self):
        """Initialize an empty dictionary to store compiled kernels."""
        self.compiled_kernels = {}

    def compile_kernel(self, kernel_code, kernel_name):
        """Compile a CUDA kernel from source code."""
        module = compiler.SourceModule(kernel_code)
        kernel = module.get_function(kernel_name)
        self.compiled_kernels[kernel_name] = kernel
        return kernel

    def launch_kernel(self, kernel, grid, block, *args):
        """Launch a CUDA kernel with the specified configuration."""
        kernel(*args, grid=grid, block=block)
