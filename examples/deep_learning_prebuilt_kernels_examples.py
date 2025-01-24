import numpy as np
from pycuda_plus.core.kernel import KernelExecutor
from pycuda_plus.core.memory import MemoryManager
from pycuda_plus.core.context import CudaContextManager
from pycuda_plus.utils.prebuilt_kernels import get_prebuilt_kernels, get_kernel_names

def test_ml_kernels():
    kernel = KernelExecutor()
    memory_manager = MemoryManager()
    context_manager = CudaContextManager()
    context_manager.initialize_context()

    try:
        prebuilt_kernels = get_prebuilt_kernels()

        for kernel_name in get_kernel_names():
            print(f"Testing kernel: {kernel_name}")
            kernel_code = prebuilt_kernels[kernel_name]
            kernel.compile_kernel(kernel_code, kernel_name)

            if kernel_name == "relu_activation":
                # Test ReLU Activation
                N = 1024
                A = np.random.randn(N).astype(np.float32)
                B = np.zeros_like(A)

                d_A = memory_manager.allocate_device_array(A.shape, dtype=np.float32)
                d_B = memory_manager.allocate_device_array(B.shape, dtype=np.float32)

                memory_manager.copy_to_device(A, d_A)

                block_size = 256
                grid_size = (N + block_size - 1) // block_size

                kernel_func = kernel.get_kernel(kernel_name)
                kernel.launch_kernel(kernel_func, (grid_size, 1, 1), (block_size, 1, 1), d_A, d_B, np.int32(N))

                memory_manager.copy_to_host(d_B, B)
                expected = np.maximum(A, 0)
                np.testing.assert_allclose(B, expected, rtol=1e-5)
                print("ReLU activation passed.")

            elif kernel_name == "leaky_relu_activation":
                # Test Leaky ReLU Activation
                N = 1024
                A = np.random.randn(N).astype(np.float32)
                alpha = 0.01
                B = np.zeros_like(A)

                d_A = memory_manager.allocate_device_array(A.shape, dtype=np.float32)
                d_B = memory_manager.allocate_device_array(B.shape, dtype=np.float32)

                memory_manager.copy_to_device(A, d_A)

                block_size = 256
                grid_size = (N + block_size - 1) // block_size

                kernel_func = kernel.get_kernel(kernel_name)
                kernel.launch_kernel(kernel_func, (grid_size, 1, 1), (block_size, 1, 1), 
                                      d_A, d_B, np.float32(alpha), np.int32(N))

                memory_manager.copy_to_host(d_B, B)
                expected = np.where(A > 0, A, alpha * A)
                np.testing.assert_allclose(B, expected, rtol=1e-5)
                print("Leaky ReLU activation passed.")

            elif kernel_name == "dropout":
                # Test Dropout
                N = 1024
                dropout_rate = 0.5
                input_data = np.random.rand(N).astype(np.float32)
                output_data = np.zeros_like(input_data)
                random_values = np.random.rand(N).astype(np.float32)

                d_input = memory_manager.allocate_device_array(input_data.shape, dtype=np.float32)
                d_output = memory_manager.allocate_device_array(output_data.shape, dtype=np.float32)
                d_random = memory_manager.allocate_device_array(random_values.shape, dtype=np.float32)

                memory_manager.copy_to_device(input_data, d_input)
                memory_manager.copy_to_device(random_values, d_random)

                block_size = 256
                grid_size = (N + block_size - 1) // block_size

                kernel_func = kernel.get_kernel(kernel_name)
                kernel.launch_kernel(kernel_func, (grid_size, 1, 1), (block_size, 1, 1), 
                                      d_input, d_output, d_random, 
                                      np.float32(dropout_rate), np.int32(N))

                memory_manager.copy_to_host(d_output, output_data)

                # Verify dropout behavior
                zeros_mask = random_values < dropout_rate
                expected_output = input_data.copy()
                expected_output[zeros_mask] = 0
                expected_output[~zeros_mask] /= (1.0 - dropout_rate)

                np.testing.assert_allclose(output_data[~zeros_mask], 
                                           expected_output[~zeros_mask], 
                                           rtol=1e-5)
                print("Dropout passed.")

        print("All ML kernel tests passed.")
    finally:
        context_manager.finalize_context()

if __name__ == "__main__":
    test_ml_kernels()
