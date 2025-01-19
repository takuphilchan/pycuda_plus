import unittest
import numpy as np
import pycuda.driver as cuda
from pycuda_plus.core.memory import MemoryManager  # For handling device memory

class TestMemoryManager(unittest.TestCase):

    def test_memory_allocation(self):
        """Test device memory allocation."""
        manager = MemoryManager(dtype=np.float32)  # Specify dtype during initialization
        shape = (10,)  # Shape of the array to allocate

        # Explicitly initialize the CUDA context
        cuda.init()
        device = cuda.Device(0)  # Choose the first GPU device
        context = device.make_context()  # Set the context

        # Allocate memory
        device_array = manager.allocate_array(shape)
        self.assertIsNotNone(device_array)  # Ensure allocation was successful
        print("Memory allocation successful")

        # Deallocate memory
        manager.deallocate(device_array)
        print("Memory deallocation successful")

        # Clean up the context after use
        context.pop()

    def test_memory_deallocation(self):
        """Test that allocated memory is correctly freed."""
        manager = MemoryManager(dtype=np.float32)
        shape = (10,)

        # Explicitly initialize the CUDA context
        cuda.init()
        device = cuda.Device(0)  # Choose the first GPU device
        context = device.make_context()  # Set the context

        # Allocate memory
        device_array = manager.allocate_array(shape)
        print(f"Allocated device memory: {device_array}")
        
        # Deallocate memory
        manager.deallocate(device_array)
        print("Memory deallocated successfully")

        # Clean up the context after use
        context.pop()

if __name__ == "__main__":
    unittest.main()
