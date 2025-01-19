import unittest
import numpy as np
from pycuda_plus.utils.numpy_support import NumpyHelper
import pycuda.gpuarray as gpuarray

class TestNumpyHelper(unittest.TestCase):
    
    def test_array_conversion(self):
        """Test NumPy to PyCUDA array conversion."""
        helper = NumpyHelper()
        
        # Create a NumPy array
        array = np.array([1, 2, 3], dtype=np.float32)
        
        # Convert to GPU array using helper's batch_copy_to_device (which uses gpuarray.to_gpu internally)
        gpu_array = helper.batch_copy_to_device([array])[0]  # batch_copy_to_device returns a list
        
        # Assert the GPU array is of the correct type
        self.assertIsInstance(gpu_array, gpuarray.GPUArray)
        
        # Check if the data in the GPU array matches the original NumPy array
        self.assertTrue(np.allclose(gpu_array.get(), array))

    def test_array_conversion_empty(self):
        """Test conversion of empty NumPy array to GPU."""
        helper = NumpyHelper()

        # Create an empty NumPy array
        array = np.array([], dtype=np.float32)

        # Convert to GPU array using helper's batch_copy_to_device
        gpu_array = helper.batch_copy_to_device([array])[0]  # batch_copy_to_device returns a list

        # Assert the GPU array is also empty
        self.assertEqual(gpu_array.size, 0)

    def test_array_back_conversion(self):
        """Test converting from GPU back to NumPy array."""
        helper = NumpyHelper()

        # Create a NumPy array
        array = np.array([1, 2, 3], dtype=np.float32)

        # Convert to GPU and back to CPU using batch_copy_to_device and batch_copy_to_host
        gpu_array = helper.batch_copy_to_device([array])[0]
        result = helper.batch_copy_to_host([gpu_array])[0]  # batch_copy_to_host returns a list

        # Check if the result matches the original array
        self.assertTrue(np.allclose(result, array))

if __name__ == "__main__":
    unittest.main()
