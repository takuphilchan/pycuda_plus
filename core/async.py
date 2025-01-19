import pycuda.driver as cuda

class AsyncExecutionManager:
    """Class to handle asynchronous kernel execution and data transfers."""
    
    def __init__(self, stream=None):
        """Initialize the execution manager with an optional CUDA stream."""
        self.stream = stream if stream else cuda.Stream()

    def execute_async(self, kernel, *args):
        """Launch a kernel asynchronously using the provided stream."""
        kernel(*args, stream=self.stream)

    def synchronize(self):
        """Synchronize the stream to ensure all operations are complete."""
        self.stream.synchronize()
    
    def __enter__(self):
        """Enable use of the context manager for asynchronous execution."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure proper cleanup of the stream (if necessary)."""
        self.stream.synchronize()  # Make sure all operations are completed

