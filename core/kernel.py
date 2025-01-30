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
        try:
            module = compiler.SourceModule(kernel_code)
            kernel = module.get_function(kernel_name)
            self.compiled_kernels[kernel_name] = kernel
            print(f"✅ Successfully compiled {kernel_name}")
            return kernel
        except Exception as e:
            print(f"❌ Kernel compilation failed for {kernel_name}: {e}")
            return None

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
        try:
            if kernel is None:
                raise ValueError("❌ Kernel is None. Ensure it is compiled successfully.")

            # ✅ Ensure CUDA context is active
            if not cuda.Context.get_current():
                raise RuntimeError("❌ CUDA context is not active! Ensure initialization.")

            # ✅ Ensure valid arguments
            for i, arg in enumerate(args):
                if arg is None:
                    raise ValueError(f"❌ Kernel argument at index {i} is None.")

            # ✅ Launch the kernel with correct argument order
            kernel(*args, block=block, grid=grid)

        except cuda.LogicError as e:
            print(f"❌ CUDA LogicError in launching kernel {str(kernel)}: {e}")
        except RuntimeError as e:
            print(f"❌ CUDA Runtime Error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error in launching kernel {str(kernel)}: {e}")
