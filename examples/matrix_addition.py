import numpy as np
from pycuda_plus.core.kernel import KernelExecutor
from pycuda_plus.core.memory import MemoryManager
from pycuda_plus.core.grid_block import GridBlockConfig
from pycuda_plus.core.profiler import PerformanceProfiler
from pycuda_plus.core.context import CudaContextManager

matrix_addition_kernel = """
__global__ void matrix_add(float *A, float *B, float *C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}
"""

def matrix_addition_with_profiling(rows, cols):
    # Initialize utilities
    kernel_executor = KernelExecutor()
    memory_manager = MemoryManager()
    grid_config = GridBlockConfig(threads_per_block=256)
    profiler = PerformanceProfiler()
    context_manager = CudaContextManager()

    context_manager.initialize_context()

    try:
        # Generate data
        A = np.random.rand(rows, cols).astype(np.float32)
        B = np.random.rand(rows, cols).astype(np.float32)
        C = np.zeros((rows, cols), dtype=np.float32)

        # Transfer data to GPU
        d_A = memory_manager.allocate_device_array(A.shape, dtype=np.float32)
        d_B = memory_manager.allocate_device_array(B.shape, dtype=np.float32)
        d_C = memory_manager.allocate_device_array(C.shape, dtype=np.float32)
        memory_manager.copy_to_device(A, d_A)
        memory_manager.copy_to_device(B, d_B)

        # Compile kernel
        compiled_kernel = kernel_executor.compile_kernel(matrix_addition_kernel, 'matrix_add')

        # Configure grid and block sizes
        total_elements = rows * cols
        grid, block = grid_config.auto_config(total_elements)

        # Adjust grid and block dimensions to 3-tuple format
        grid = (grid[0], grid[0], 1)  # 2D grid for matrix operations
        block = (block[0], 1, 1)  # 1D blocks for threads

        # Profile kernel execution
        execution_time = profiler.profile_kernel(
            compiled_kernel, grid, block, d_A, d_B, d_C, np.int32(rows), np.int32(cols)
        )
        print(f"Matrix addition kernel execution time: {execution_time:.6f} seconds")

        # Copy result back to host
        memory_manager.copy_to_host(d_C, C)

        return C

    finally:
        # Clean up CUDA context
        context_manager.finalize_context()

if __name__ == "__main__":
    rows, cols = 1024, 1024  # Matrix dimensions
    result = matrix_addition_with_profiling(rows, cols)
    print(f"Matrix addition result (first 5x5 elements):\n{result[:5, :5]}")
