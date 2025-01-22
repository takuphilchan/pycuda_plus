import numpy as np
from pycuda_plus.utils.visualization import plot_memory_usage, plot_execution_times, real_time_memory_monitor
from pycuda_plus.core.kernel import KernelExecutor
from pycuda_plus.core.memory import MemoryManager
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

matrix_multiplication_kernel = """
__global__ void matrix_mul(float *A, float *B, float *C, int rows, int cols) {
    __shared__ float sA[16][16];
    __shared__ float sB[16][16];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0;

    for (int tile = 0; tile < (cols + 16 - 1) / 16; ++tile) {
        if (row < rows && tile * 16 + threadIdx.x < cols)
            sA[threadIdx.y][threadIdx.x] = A[row * cols + tile * 16 + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0;

        if (col < cols && tile * 16 + threadIdx.y < rows)
            sB[threadIdx.y][threadIdx.x] = B[(tile * 16 + threadIdx.y) * cols + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int k = 0; k < 16; ++k)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < rows && col < cols)
        C[row * cols + col] = sum;
}
"""

def get_memory_usage(memory_manager):
    """
    Get the current memory usage on the GPU.
    """
    total_memory, used_memory = memory_manager.get_memory_info()
    return total_memory, used_memory

def compare_kernels(rows, cols):
    """
    Compare the execution times of matrix addition and multiplication kernels.

    Parameters:
        rows (int): Number of rows in the matrix.
        cols (int): Number of columns in the matrix.
    """
    # Initialize utilities
    kernel_executor = KernelExecutor()
    memory_manager = MemoryManager()
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

        # Visualize memory usage (initial snapshot)
        total_memory, used_memory = memory_manager.get_memory_info()
        plot_memory_usage(total_memory, used_memory)

        # Start real-time memory monitoring in a background thread
        # Monitor memory usage for the entire duration of kernel execution
        real_time_memory_monitor(lambda: get_memory_usage(memory_manager), duration=10, interval=1)

        # Compile kernels
        add_kernel = kernel_executor.compile_kernel(matrix_addition_kernel, 'matrix_add')
        mul_kernel = kernel_executor.compile_kernel(matrix_multiplication_kernel, 'matrix_mul')

        # Configure grid and block sizes
        block = (16, 16, 1)  # 16x16 threads per block
        grid = ((cols + block[0] - 1) // block[0], (rows + block[1] - 1) // block[1], 1)

        # Profile addition kernel
        execution_times = {"Addition": []}
        for _ in range(5):  # Run multiple times for averaging
            execution_time = profiler.profile_kernel(
                add_kernel, grid, block, d_A, d_B, d_C, np.int32(rows), np.int32(cols)
            )
            execution_times["Addition"].append(execution_time)

        # Profile multiplication kernel
        execution_times["Multiplication"] = []
        for _ in range(5):
            execution_time = profiler.profile_kernel(
                mul_kernel, grid, block, d_A, d_B, d_C, np.int32(rows), np.int32(cols)
            )
            execution_times["Multiplication"].append(execution_time)

        # Visualize kernel execution times
        plot_execution_times(execution_times)

        # Copy result back to host for verification (optional)
        memory_manager.copy_to_host(d_C, C)

        return C

    finally:
        # Clean up CUDA context
        context_manager.finalize_context()

if __name__ == "__main__":
    rows, cols = 2024, 2024  # Example size
    result = compare_kernels(rows, cols)
    print(f"Comparison completed. Final result (first 5x5 elements):\n{result[:5, :5]}")
