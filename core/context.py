import pycuda.driver as cuda
import pycuda.autoinit  # This is important to initialize CUDA
import numpy as np

class CudaContextManager:
    """Class to manage CUDA context initialization and finalization."""
    
    def __init__(self, device_id=0):
        """Initialize CUDA context for the specified device."""
        self.device_id = device_id
        self.context = None
    
    def initialize_context(self):
        """Initialize a CUDA context on the specified device."""
        cuda.init()  # Initialize the CUDA driver
        device = cuda.Device(self.device_id)  # Select the device
        self.context = device.make_context()  # Create context for the device
        print(f"CUDA context initialized for device {self.device_id}.")
        return self.context
    
    def finalize_context(self):
        """Clean up the CUDA context."""
        if self.context:
            self.context.pop()  # Pop the context off the stack
            del self.context  # Delete the context
            print(f"CUDA context for device {self.device_id} finalized.")
        else:
            print("No context to finalize.")
    
    def __enter__(self):
        """Enable use of the context manager with 'with' statement."""
        return self.initialize_context()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and finalize resources."""
        self.finalize_context()

    def is_context_active(self):
        """Check if the context is active."""
        return hasattr(self, 'context') and self.context is not None
