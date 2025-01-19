import time
import pycuda.driver as cuda

class PerformanceProfiler:
    """Provides performance profiling tools for CUDA kernel execution."""

    def __init__(self):
        """Initialize the profiler."""
        pass

    def profile_kernel(self, kernel, grid, block, *args):
        """Time the execution of a CUDA kernel."""
        start = time.time()
        kernel(*args, grid=grid, block=block)
        cuda.Context.synchronize()  # Ensure the kernel finishes execution before stopping the timer
        end = time.time()
        return end - start
