import atexit
import pycuda.driver as cuda

class CudaContextManager:
    """Class to manage CUDA context initialization and finalization.""" 
    
    def __init__(self, device_id=0):
        """Initialize CUDA context for the specified device.""" 
        self.device_id = device_id 
        self.context = None 
        atexit.register(self.finalize_context)  # Ensure cleanup on program exit

    def initialize_context(self):
        """Initialize a CUDA context on the specified device.""" 
        if not self.is_context_active():  # Ensure context is not already initialized
            cuda.init()  # Initialize the CUDA driver
            device = cuda.Device(self.device_id)  # Select the device
            self.context = device.make_context()  # Create context for the device
            print(f"CUDA context initialized for device {self.device_id}.")
        else:
            print(f"CUDA context already initialized for device {self.device_id}.")
        return self.context
    
    def finalize_context(self):
        """Clean up the CUDA context.""" 
        if self.is_context_active():  # Only finalize if context is active
            try:
                self.context.pop()  # Pop context
                self.context.detach()  # Detach context to fully clean it up
                del self.context  # Delete the context reference
                self.context = None
                print(f"CUDA context for device {self.device_id} finalized.")
            except cuda.LogicError:
                print("CUDA context was already inactive or deinitialized.")
        else:
            print(f"No active context for device {self.device_id} to finalize.")

    def is_context_active(self):
        """Check if the context is active.""" 
        return self.context is not None

    def __enter__(self):
        """Enable use of the context manager with 'with' statement.""" 
        return self.initialize_context()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and finalize resources."""
        self.finalize_context()
        if exc_type is not None:
            print(f"An exception occurred: {exc_type}, {exc_val}")
        return False
