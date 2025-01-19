import unittest
import numpy as np
from pycuda_plus.core.kernel import KernelExecutor
from pycuda_plus.core.memory import MemoryManager  # For handling device memory
import pycuda.driver as cuda
import pycuda.autoinit  # This is essential for automatic initialization of CUDA

class TestKernelExecutor(unittest.TestCase):
    def test_kernel_execution(self):
        """Test a simple kernel execution to add two vectors."""

        # Initialize the kernel executor and memory manager
        executor = KernelExecutor()
        memory_manager = MemoryManager()  # For allocating and transferring data

        # Define the kernel code for adding two vectors
        kernel_code = """
        __global__ void add(float *a, float *b, float *c, int n) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx < n) {
                c[idx] = a[idx] + b[idx];
            }
        }
        """
        
        # Compile the kernel and get the function
        try:
            add_kernel = executor.compile_kernel(kernel_code, "add")
            print("Kernel compiled successfully.")
        except Exception as e:
            self.fail(f"Kernel compilation failed: {e}")

        # Prepare data for the test
        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array([4, 5, 6], dtype=np.float32)
        c = np.zeros_like(a)

        # Allocate device memory and copy data to GPU
        d_a = memory_manager.allocate_array_like(a)
        d_b = memory_manager.allocate_array_like(b)
        d_c = memory_manager.allocate_array_like(c)
        
        try:
            memory_manager.copy_to_device(a, d_a)
            memory_manager.copy_to_device(b, d_b)
            print("Data copied to device.")
        except Exception as e:
            self.fail(f"Data copy to device failed: {e}")

        # Set the grid and block size
        grid = (1, 1)  # One block
        block = (len(a), 1, 1)  # Set block size as (x, y, z)

        # Execute the kernel to add vectors a and b
        try:
            executor.launch_kernel(add_kernel, grid, block, d_a, d_b, d_c, np.int32(len(a)))
            cuda.Context.synchronize()  # Ensure kernel execution is finished
            print("Kernel execution completed.")
        except Exception as e:
            self.fail(f"Kernel execution failed: {e}")

        # Copy the result back from device to host
        try:
            memory_manager.copy_to_host(d_c, c)
            print("Data copied back to host.")
        except Exception as e:
            self.fail(f"Data copy back to host failed: {e}")

        # Check if the result is correct
        self.assertTrue(np.allclose(c, a + b), f"Result mismatch: {c} != {a + b}")

if __name__ == "__main__":
    unittest.main()
