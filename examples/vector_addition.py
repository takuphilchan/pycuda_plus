import numpy as np
from pycuda_plus.core.kernel import KernelExecutor
from pycuda_plus.core.memory import MemoryManager
from pycuda_plus.utils.prebuilt_kernels import get_kernel
from pycuda_plus.core.error import CudaErrorChecker  # Using class-based error checking
from pycuda_plus.core.context import CudaContextManager  # Using class-based context management

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
    N = 1000000  # Size of the vectors
    result = vector_addition_example(N)
    if result is not None:
        print(f"Vector addition result (first 5 elements):\n{result[:5]}")
    else:
        print("Error in vector addition.")
