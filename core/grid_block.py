class GridBlockConfig:
    """Class to automate grid and block configuration for CUDA kernels."""

    def __init__(self, threads_per_block=256):
        """Initialize the configuration with the number of threads per block."""
        self.threads_per_block = threads_per_block

    def auto_config(self, shape):
        """Calculate grid and block sizes based on input shape."""
        total_threads = shape[0] if isinstance(shape, (tuple, list)) else shape
        blocks = (total_threads + self.threads_per_block - 1) // self.threads_per_block
        return (blocks, 1), (self.threads_per_block, 1)
