from pycuda_plus.core.memory import MemoryManager
import numpy as np
from pycuda_plus.core.context import CudaContextManager  # Using class-based context management

manager = MemoryManager()
context_manager = CudaContextManager()

# Initialize CUDA context once at the beginning
context_manager.initialize_context()

# NumPy to GPU
np_array = np.random.rand(10, 10).astype(np.float32)
device_array = manager.numpy_to_device(np_array)

# GPU to NumPy
result_np = manager.device_to_numpy(device_array, shape=(10, 10), dtype=np.float32)
print("✅ Success: Result NumPy Array:\n", result_np)

# Initialize CuPy Array
cp_array = manager.numpy_to_cupy(np_array)
print("✅ Success: CuPy Array:\n", cp_array)

# CuPy to GPU
device_from_cupy = manager.cupy_to_device(cp_array)

# GPU to CuPy
cp_result = manager.device_to_cupy(device_from_cupy, shape=(10, 10), dtype=np.float32)
print("✅ Success: Result CuPy Array:\n", cp_result)

# CuPy to NumPy
final_np = manager.cupy_to_numpy(cp_result)
print("✅ Success: Final NumPy Array:\n", final_np)

# Finalize the context at the end
context_manager.finalize_context()
