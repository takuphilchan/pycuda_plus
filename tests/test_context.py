import unittest
from pycuda_plus.core.context import CudaContextManager  # Import the updated context manager class

class TestCudaContextManager(unittest.TestCase):
    def test_context_creation(self):
        """Test that CUDA context is created and released without errors."""
        context_manager = CudaContextManager()
        
        # Using the context manager in the 'with' block
        with context_manager:
            self.assertTrue(context_manager.is_context_active())  # Check if the context is active

        self.assertFalse(context_manager.is_context_active())  # Check if the context is inactive after cleanup

    def test_context_finalization(self):
        """Test the finalization of the CUDA context."""
        context_manager = CudaContextManager()
        
        # Ensure context is initialized before finalization
        context_manager.initialize_context()
        
        # Ensure context is active before finalization
        self.assertTrue(context_manager.is_context_active())
        
        # Finalize the context
        context_manager.finalize_context()
        
        # After finalization, ensure context is no longer active
        self.assertFalse(context_manager.is_context_active())

if __name__ == "__main__":
    unittest.main()
