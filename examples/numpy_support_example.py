import numpy as np
from pycuda_plus.core.memory import MemoryManager
from pycuda_plus.utils.numpy_support import NumpyHelper
from pycuda_plus.core.context import CudaContextManager

def example_using_numpy_helper(N):
    # Instantiate required components
    memory_manager = MemoryManager()
    numpy_helper = NumpyHelper()  # We'll keep this in case we need other helper functions
    context_manager = CudaContextManager()

    # Initialize the CUDA context
    context_manager.initialize_context()

    try:
        # Create an array on the host
        host_array1 = np.random.rand(N).astype(np.float32)
        host_array2 = np.random.rand(N).astype(np.float32)

        # Generate a patterned array using NumpyHelper (e.g., a range array)
        d_patterned_array = numpy_helper.generate_patterned_array((N,), 'range')
        patterned_array = numpy_helper.batch_copy_to_host([d_patterned_array])[0]

        # Batch copy arrays to device memory using NumpyHelper
        d_array1, d_array2 = numpy_helper.batch_copy_to_device([host_array1, host_array2])

        # Print some results
        print("Patterned array (first 10 elements):", patterned_array[:10])
        print("Host Array 1 (first 10 elements):", host_array1[:10])
        print("Host Array 2 (first 10 elements):", host_array2[:10])

        # Return the arrays for further use if needed
        return {
            "patterned_array": patterned_array[:10],
            "host_array1": host_array1[:10],
            "host_array2": host_array2[:10],
        }

    finally:
        # Finalize CUDA context
        context_manager.finalize_context()

if __name__ == "__main__":
    N = 10000  # Array size
    results = example_using_numpy_helper(N)

    # Print results
    print("Patterned array (first 10 elements):", results["patterned_array"])
    print("Host Array 1 (first 10 elements):", results["host_array1"])
    print("Host Array 2 (first 10 elements):", results["host_array2"])
