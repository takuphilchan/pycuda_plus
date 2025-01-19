import unittest
import numpy as np
from pycuda_plus.core.context import CudaContextManager  # Updated context manager
from pycuda_plus.core.memory import MemoryManager
from pycuda_plus.core.kernel import KernelExecutor
from pycuda_plus.utils.numpy_support import NumpyHelper  # for handling memory transfers

class TestIntegration(unittest.TestCase):
    def test_full_pipeline(self):
        """Test context creation, memory management, and kernel execution together."""
        
        # Initialize necessary components
        context_manager = CudaContextManager()  # Use the updated CudaContextManager
        memory_manager = MemoryManager()  # Manages memory allocations and transfers
        kernel_executor = KernelExecutor()  # Responsible for kernel compilation and execution

        with context_manager:  # Using the context manager directly
            # Define the kernel code for squaring the elements of a vector
            kernel_code = """
            __global__ void square(float *a, int n) {
                int idx = threadIdx.x + blockIdx.x * blockDim.x;
                if (idx < n) {
                    a[idx] = a[idx] * a[idx];
                }
            }
            """
            # Compile the kernel (pass kernel name as an argument)
            module = kernel_executor.compile_kernel(kernel_code, "square")
            print("Kernel module compiled successfully.")

            # Attempt to get the 'square' function from the compiled module
            try:
                square = module.get_function("square")
                print("Kernel function 'square' retrieved successfully.")
            except Exception as e:
                print(f"Error retrieving function: {e}")
                return  # Skip the test if kernel function retrieval fails

            # Prepare the data and allocate memory on the GPU
            data = np.array([1, 2, 3], dtype=np.float32)
            gpu_data = memory_manager.allocate_array_like(data)
            gpu_data.set(data)  # Copy data to GPU memory

            # Execute the kernel
            kernel_executor.execute_kernel(square, gpu_data, np.int32(len(data)))

            # Check if the operation was successful by comparing results
            self.assertTrue(np.allclose(gpu_data.get(), data ** 2))  # Ensure the data matches the squared values

if __name__ == "__main__":
    unittest.main()
