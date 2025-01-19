import numpy as np
from pycuda_plus.core.kernel import KernelExecutor
from pycuda_plus.core.memory import MemoryManager
from pycuda_plus.utils.numpy_support import NumpyHelper
from pycuda_plus.core.error import CudaErrorChecker  # Using class-based error checking
from pycuda_plus.core.context import CudaContextManager  # Using class-based context management

# Custom kernel for squaring elements of a vector
custom_kernel_code = """
__global__ void square_elements(float *a, float *b, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        b[idx] = a[idx] * a[idx];
    }
}
"""

def custom_kernel_example(N):
    kernel = KernelExecutor()
    memory = MemoryManager()
    numpy_helper = NumpyHelper()
    # Initialize CUDA context
    context_manager = CudaContextManager()  # Using class-based context management
    context_manager.initialize_context()  # Using context from context.py
    
    try:
        # Generate random vector
        A = np.random.rand(N).astype(np.float32)
        B = np.zeros(N, dtype=np.float32)

        # Compile the kernel
        square_elements = kernel.compile_kernel(custom_kernel_code, 'square_elements')

        # Allocate device memory
        d_A = memory.allocate_device_array(A.shape, dtype=np.float32)
        d_B = memory.allocate_device_array(B.shape, dtype=np.float32)

        # Copy data to device
        memory.copy_to_device(A, d_A)

        # Configure grid and block dimensions
        block_size = 256
        grid_size = (N + block_size - 1) // block_size  # Calculate grid size

        # Launch the kernel with a 3D block configuration
        block = (block_size, 1, 1)  # (block_size, 1, 1) for 1D thread block
        grid = (grid_size, 1, 1)   # (grid_size, 1, 1) for 1D grid
        
        # Launch the kernel
        kernel.launch_kernel(square_elements, grid, block, d_A, d_B, np.int32(N))

        # Synchronize and check for errors using CudaErrorChecker
        error_checker = CudaErrorChecker()
        error_checker.check_errors()

        # Copy the result back to the host
        memory.copy_to_host(d_B, B)
        return B

    finally:
        # Finalize CUDA context
        context_manager.finalize_context()  # Clean up context after computation

if __name__ == "__main__":
    N = 1000000  # Size of the vector
    result = custom_kernel_example(N)
    print(f"Custom kernel result (first 5 elements):\n{result[:5]}")
