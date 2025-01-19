import pycuda.driver as cuda

class CudaErrorChecker:
    """Handles checking for CUDA errors."""

    def __init__(self):
        """Initialize the CUDA error checker."""
        pass

    def check_errors(self):
        """Check for CUDA errors."""
        try:
            # Force synchronization to catch any CUDA errors
            cuda.Context.synchronize()
        except cuda.Error as e:
            raise RuntimeError(f"CUDA error: {e}")
