import numpy as np
import traceback
from pycuda_plus.core.kernel import KernelExecutor
from pycuda_plus.core.memory import MemoryManager
from pycuda_plus.core.profiler import PerformanceProfiler
from pycuda_plus.core.context import CudaContextManager
from pycuda_plus.utils.visualization import plot_execution_times
from pycuda_plus.core.grid_block import GridBlockConfig
from pycuda_plus.utils.prebuilt_kernels import get_prebuilt_kernels

class GpuKernelComparator:
    def __init__(self):
        self.kernel_executor = KernelExecutor()
        self.memory_manager = MemoryManager()
        self.profiler = PerformanceProfiler()
        self.context_manager = CudaContextManager()
        self.grid_config = GridBlockConfig(threads_per_block=256)

    def compare_kernels(self, rows, cols, operations=None):
        execution_times = {}
        try:
            # Initialize CUDA context
            print("Initializing CUDA context...")
            self.context_manager.initialize_context()
            print("CUDA context initialized.")

            # Allocate matrices
            A = np.random.rand(rows, cols).astype(np.float32)
            B = np.random.rand(rows, cols).astype(np.float32)
            C = np.zeros((rows, cols), dtype=np.float32)

            # Allocate memory on the GPU
            print("Allocating memory on the device...")
            d_A = self.memory_manager.numpy_to_device(A)
            d_B = self.memory_manager.numpy_to_device(B)
            d_C = self.memory_manager.allocate_device_array(C.shape, C.dtype)
            print("Memory allocation successful.")

            # Retrieve prebuilt kernels
            prebuilt_kernels = get_prebuilt_kernels()

            # Get the 'vector_scale' and 'matrix_multiply' kernel code from prebuilt kernels
            matrix_add_kernel_code = prebuilt_kernels['matrix_add']
            matrix_multiply_kernel_code = prebuilt_kernels['matrix_multiply']

            # Compile the kernels
            matrix_add_compiled_kernel = self.kernel_executor.compile_kernel(matrix_add_kernel_code, "matrix_add")
            matrix_multiply_compiled_kernel = self.kernel_executor.compile_kernel(matrix_multiply_kernel_code, "matrix_multiply")
            print("Kernels compiled.")

            # Define block and grid size
            block = (16, 16, 1)  # 16x16 threads per block for matrix multiplication
            grid = ((cols + block[0] - 1) // block[0], (rows + block[1] - 1) // block[1], 1)
            print(f"Block size: {block}, Grid size: {grid}")

            # Run the vector_scale kernel if requested or if no specific operation is given
            if operations is None or "matrix_add" in operations:
                try:
                    print("Launching kernel: matrix_add")
                    exec_time = self.profiler.profile_kernel(
                        matrix_add_compiled_kernel, grid, block, d_A, d_B, d_C, np.int32(rows), np.int32(cols)
                    )
                    execution_times["Matrix Addition"] = [exec_time]  # Ensure it's a list
                    print("Vector scale kernel executed successfully.")
                except Exception as e:
                    print("Error launching Vector Scale kernel:")
                    traceback.print_exc()

            # Run the matrix_multiply kernel if requested or if no specific operation is given
            if operations is None or "matrix_multiply" in operations:
                try:
                    print("Launching kernel: matrix_multiply")
                    exec_time = self.profiler.profile_kernel(
                        matrix_multiply_compiled_kernel, grid, block, d_A, d_B, d_C, np.int32(rows), np.int32(cols)
                    )
                    execution_times["Matrix Multiply"] = [exec_time]  # Ensure it's a list
                    print("Matrix multiplication kernel executed successfully.")
                except Exception as e:
                    print("Error launching Matrix Multiply kernel:", e)
                    traceback.print_exc()

            # Deallocate memory
            print("🗑️ Deallocating memory...")
            self.memory_manager.deallocate(d_A, d_B, d_C)
            print("Memory deallocated.")

        except Exception as e:
            print("General error in kernel execution:")
            traceback.print_exc()  # Print full stack trace
            return None

        finally:
            print("Finalizing CUDA context...")
            self.context_manager.finalize_context()
            print("CUDA context finalized.")

        return execution_times if execution_times else None

if __name__ == "__main__":
    try:
        kernel_comparator = GpuKernelComparator()
        rows, cols = 10, 10
        plot_execution_times(kernel_comparator=kernel_comparator, rows=rows, cols=cols)  # Plot execution times
    except Exception as e:
        print(f"Error during execution: {e}")
