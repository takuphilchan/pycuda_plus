import numpy as np
from pycuda_plus.core.kernel import KernelExecutor
from pycuda_plus.core.memory import MemoryManager 
from pycuda_plus.core.grid_block import GridBlockConfig
from pycuda_plus.utils import numpy_support
from pycuda_plus.core.context import CudaContextManager  # Using class-based context management
from pycuda_plus.core.error import CudaErrorChecker  # Using class-based error checking

# CUDA C kernel for matrix multiplication
matrix_multiply_kernel = """
__global__ void matrix_multiply(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float value = 0;
        for (int k = 0; k < N; ++k) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}
"""

def matrix_multiply_example(N):
    kernel = KernelExecutor()
    memory = MemoryManager()
    # Initialize CUDA context
    context_manager = CudaContextManager()  # Using class-based context management
    context_manager.initialize_context()  # Using context from context.py
    
    try:
        # Generate random matrices
        A = np.random.rand(N, N).astype(np.float32)
        B = np.random.rand(N, N).astype(np.float32)
        C = np.zeros((N, N), dtype=np.float32)

        # Compile the kernel
        matrix_multiply = kernel.compile_kernel(matrix_multiply_kernel, 'matrix_multiply')

        # Allocate device memory
        d_A = memory.allocate_device_array(A.shape, dtype=np.float32)
        d_B = memory.allocate_device_array(B.shape, dtype=np.float32)
        d_C = memory.allocate_device_array(C.shape, dtype=np.float32)

        # Copy data to device
        memory.copy_to_device(A, d_A)
        memory.copy_to_device(B, d_B)

        # Configure grid and block dimensions
        block_size = 16
        grid = ((N + block_size - 1) // block_size, (N + block_size - 1) // block_size)
        block = (block_size, block_size, 1)  # Add z-dimension for block size

        # Launch the kernel
        kernel.launch_kernel(matrix_multiply, grid, block, d_A, d_B, d_C, np.int32(N))

        # Synchronize and check for errors using CudaErrorChecker
        error_checker = CudaErrorChecker()
        error_checker.check_errors()

        # Copy the result back to the host
        memory.copy_to_host(d_C, C)
        return C

    finally:
        # Finalize the CUDA context
        context_manager.finalize_context()  # No need to pass context here

if __name__ == "__main__":
    N = 512  # Size of the matrices
    result = matrix_multiply_example(N)
    print(f"Matrix multiplication result (first 5 elements):\n{result[:5, :5]}")
