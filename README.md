# PyCUDA Plus

**PyCUDA Plus** is an enhanced Python library built on top of PyCUDA, designed to simplify GPU programming and execution. It provides high-level abstractions and utilities for working with CUDA kernels, memory management, and context handling, allowing developers to focus on writing efficient CUDA code without dealing with low-level details.

---

## Key Features

- **Kernel Management:** Compile, load, and execute custom CUDA kernels easily with the `KernelExecutor`.
- **Memory Management:** Simplified allocation and transfer of device and host memory using the `MemoryManager`.
- **Context Handling:** Seamless setup and teardown of CUDA contexts with the `CudaContextManager`.
- **Error Checking:** Built-in error detection and reporting via `CudaErrorChecker`.
- **Utility Functions:** Prebuilt kernels, NumPy support, and grid/block configuration helpers for common operations.
- **Grid/Block Configuration:** Automate grid and block size calculations for CUDA kernels using `GridBlockConfig`.
- **Performance Profiling:** Measure execution time of CUDA kernels with `PerformanceProfiler`.

---

## Installation

To install the `pycuda_plus` library, run:

```bash
pip install pycuda_plus
```

Ensure you have the following prerequisites installed:
- CUDA Toolkit
- PyCUDA
- Compatible NVIDIA GPU drivers

---

## Getting Started

### Example 1: Vector Addition

```python
import numpy as np
from pycuda_plus.core.kernel import KernelExecutor
from pycuda_plus.core.memory import MemoryManager
from pycuda_plus.utils.prebuilt_kernels import get_kernel
from pycuda_plus.core.context import CudaContextManager
from pycuda_plus.core.error import CudaErrorChecker


def vector_addition_example(N):
    kernel = KernelExecutor()
    memory_manager = MemoryManager()  # Using the MemoryManager
    context_manager = CudaContextManager()
    context_manager.initialize_context()

    try:
        A = np.random.rand(N).astype(np.float32)
        B = np.random.rand(N).astype(np.float32)
        C = np.zeros(N, dtype=np.float32)

        vector_add = get_kernel('vector_add')

        # Allocate memory on the GPU
        d_A = memory_manager.allocate_device_array(A.shape, dtype=np.float32)
        d_B = memory_manager.allocate_device_array(B.shape, dtype=np.float32)
        d_C = memory_manager.allocate_device_array(C.shape, dtype=np.float32)

        # Copy data from host to GPU
        memory_manager.copy_to_device(A, d_A)
        memory_manager.copy_to_device(B, d_B)

        block_size = 256
        grid_size = (N + block_size - 1) // block_size

        # Launch the kernel
        kernel.launch_kernel(vector_add, (grid_size, 1, 1), (block_size, 1, 1), d_A, d_B, d_C, np.int32(N))

        error_checker = CudaErrorChecker()
        error_checker.check_errors()

        # Copy the result back to host
        memory_manager.copy_to_host(d_C, C)
        return C
    finally:
        context_manager.finalize_context()


if __name__ == "__main__":
    N = 1000000
    result = vector_addition_example(N)
    print(f"Vector addition result (first 5 elements): {result[:5]}")
```

### Example 2: Matrix Multiplication

```python
import numpy as np
from pycuda_plus.core.kernel import KernelExecutor
from pycuda_plus.core.memory import MemoryManager
from pycuda_plus.core.context import CudaContextManager
from pycuda_plus.core.error import CudaErrorChecker

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
    memory_manager = MemoryManager()
    context_manager = CudaContextManager()
    context_manager.initialize_context()

    try:
        # Host arrays
        A = np.random.rand(N, N).astype(np.float32)
        B = np.random.rand(N, N).astype(np.float32)
        C = np.zeros((N, N), dtype=np.float32)

        # Compile the kernel
        compiled_kernel = kernel.compile_kernel(matrix_multiply_kernel, 'matrix_multiply')

        # Allocate memory on the device
        d_A = memory_manager.allocate_device_array(A.shape, dtype=np.float32)
        d_B = memory_manager.allocate_device_array(B.shape, dtype=np.float32)
        d_C = memory_manager.allocate_device_array(C.shape, dtype=np.float32)

        # Copy data to device
        memory_manager.copy_to_device(A, d_A)
        memory_manager.copy_to_device(B, d_B)

        # Configure grid and block sizes
        block_size = 16
        grid_size = (N + block_size - 1) // block_size

        # Launch the kernel
        kernel.launch_kernel(
            compiled_kernel,
            (grid_size, grid_size, 1),
            (block_size, block_size, 1),
            d_A, d_B, d_C, np.int32(N)
        )

        # Error checking
        error_checker = CudaErrorChecker()
        error_checker.check_errors()

        # Copy the result back to the host
        memory_manager.copy_to_host(d_C, C)
        return C

    finally:
        # Finalize the context
        context_manager.finalize_context()

if __name__ == "__main__":
    N = 512
    result = matrix_multiply_example(N)
    print(f"Matrix multiplication result (first 5x5 elements):\n{result[:5, :5]}")
```

### Example 3: Matrix Addition with Profiling

```python
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
    kernel_executor = KernelExecutor()
    memory_manager = MemoryManager()
    grid_config = GridBlockConfig(threads_per_block=256)
    profiler = PerformanceProfiler()
    context_manager = CudaContextManager()

    context_manager.initialize_context()

    try:
        A = np.random.rand(rows, cols).astype(np.float32)
        B = np.random.rand(rows, cols).astype(np.float32)
        C = np.zeros((rows, cols), dtype=np.float32)

        d_A = memory_manager.allocate_device_array(A.shape, dtype=np.float32)
        d_B = memory_manager.allocate_device_array(B.shape, dtype=np.float32)
        d_C = memory_manager.allocate_device_array(C.shape, dtype=np.float32)
        memory_manager.copy_to_device(A, d_A)
        memory_manager.copy_to_device(B, d_B)

        compiled_kernel = kernel_executor.compile_kernel(matrix_addition_kernel, 'matrix_add')

        total_elements = rows * cols
        grid, block = grid_config.auto_config(total_elements)

        grid = (grid[0], grid[0], 1)
        block = (block[0], 1, 1)

        execution_time = profiler.profile_kernel(
            compiled_kernel, grid, block, d_A, d_B, d_C, np.int32(rows), np.int32(cols)
        )
        print(f"Matrix addition kernel execution time: {execution_time:.6f} seconds")

        memory_manager.copy_to_host(d_C, C)

        return C

    finally:
        context_manager.finalize_context()

if __name__ == "__main__":
    rows, cols = 1024, 1024
    result = matrix_addition_with_profiling(rows, cols)
    print(f"Matrix addition result (first 5x5 elements):\n{result[:5, :5]}")
```

### Example 4: Numpy Helper

```python
import numpy as np
from pycuda_plus.core.memory import MemoryManager
from pycuda_plus.utils.numpy_support import NumpyHelper
from pycuda_plus.core.context import CudaContextManager

def example_using_numpy_helper(N):
    # Instantiate required components
    memory_manager = MemoryManager()
    numpy_helper = NumpyHelper()  # We'll keep this in case we need other helper functions
    context_manager = CudaContextManager()

    # Initialize the CUDA context
    context_manager.initialize_context()

    try:
        # Create an array on the host
        host_array1 = np.random.rand(N).astype(np.float32)
        host_array2 = np.random.rand(N).astype(np.float32)

        # Generate a patterned array using NumpyHelper (e.g., a range array)
        d_patterned_array = numpy_helper.generate_patterned_array((N,), 'range')
        patterned_array = numpy_helper.batch_copy_to_host([d_patterned_array])[0]

        # Batch copy arrays to device memory using NumpyHelper
        d_array1, d_array2 = numpy_helper.batch_copy_to_device([host_array1, host_array2])

        # Print some results
        print("Patterned array (first 10 elements):", patterned_array[:10])
        print("Host Array 1 (first 10 elements):", host_array1[:10])
        print("Host Array 2 (first 10 elements):", host_array2[:10])

        # Return the arrays for further use if needed
        return {
            "patterned_array": patterned_array[:10],
            "host_array1": host_array1[:10],
            "host_array2": host_array2[:10],
        }

    finally:
        # Finalize CUDA context
        context_manager.finalize_context()

if __name__ == "__main__":
    N = 10000  # Array size
    results = example_using_numpy_helper(N)

    # Print results
    print("Patterned array (first 10 elements):", results["patterned_array"])
    print("Host Array 1 (first 10 elements):", results["host_array1"])
    print("Host Array 2 (first 10 elements):", results["host_array2"])

```


---

## API Documentation

### Core Modules

1. **`KernelExecutor`**
   - Compile and launch CUDA kernels.
   - Example:
     ```python
     kernel_executor = KernelExecutor()
     compiled_kernel = kernel_executor.compile_kernel(kernel_code, kernel_name)
     kernel_executor.launch_kernel(compiled_kernel, grid, block, *args)
     ```

2. **`MemoryManager`**
   - Allocate, manage, and transfer memory between host and device.
   - Example:
     ```python
     memory_manager = MemoryManager()
     device_array = memory_manager.allocate_device_array(shape, dtype)
     memory_manager.copy_to_device(host_array, device_array)
     memory_manager.copy_to_host(device_array, host_array)
     ```

3. **`CudaContextManager`**
   - Simplify CUDA context setup and teardown.
   - Example:
     ```python
     context_manager = CudaContextManager()
     context_manager.initialize_context()
     context_manager.finalize_context()
     ```

4. **`CudaErrorChecker`**
   - Check for CUDA errors during kernel execution.
   - Example:
     ```python
     error_checker = CudaErrorChecker()
     error_checker.check_errors()
     ```

5. **`GridBlockConfig`**
   - Automate grid and block size calculation.
   - Example:
     ```python
     grid_config = GridBlockConfig(threads_per_block=256)
     grid, block = grid_config.auto_config(shape)
     print(f"Grid: {grid}, Block: {block}")
     ```

6. **`PerformanceProfiler`**
   - Measure execution time of CUDA kernels.
   - Example:
     ```python
     profiler = PerformanceProfiler()
     execution_time = profiler.profile_kernel(kernel, grid, block, *args)
     print(f"Kernel execution time: {execution_time:.6f} seconds")
     ```
## 7. **`NumpyHelper`**
   - **Purpose**: Provide advanced utilities for integrating NumPy arrays with CUDA device memory using `pycuda_plus`.
   - **Functions**:
     - **`reshape_device_array(device_array, new_shape)`**: Reshape a device array into a new shape without changing its contents.
     - **`generate_patterned_array(shape, pattern)`**: Generate patterned device arrays (e.g., `range`, `linspace`) for device-side operations.
     - **`batch_copy_to_device(numpy_arrays)`**: Batch copy multiple NumPy arrays to device memory.
     - **`batch_copy_to_host(device_arrays)`**: Batch copy multiple device arrays to host memory.
   - **Example**:
     ```python
     numpy_helper = NumpyHelper()
     d_patterned_array = numpy_helper.generate_patterned_array((1000,), 'range')
     d_array1, d_array2 = numpy_helper.batch_copy_to_device([array1, array2])
     result = numpy_helper.batch_copy_to_host([d_result])[0]
     ```

### Utility Modules

- **`numpy_support`**: Convert between NumPy arrays and GPU memory.
- **`prebuilt_kernels`**: Access commonly used CUDA kernels.
- **`grid_block`**: Helpers for calculating grid and block dimensions.
- **`profiler`**: Tools for profiling CUDA kernel execution.

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests on the [GitHub repository](https://github.com/your-repo-link).

---

## License

PyCUDA Plus is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Built on the foundation of PyCUDA, with additional utilities for enhanced usability and performance.

