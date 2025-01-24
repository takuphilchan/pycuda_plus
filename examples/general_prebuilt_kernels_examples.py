import numpy as np
from pycuda_plus.core.kernel import KernelExecutor
from pycuda_plus.core.memory import MemoryManager
from pycuda_plus.core.error import CudaErrorChecker
from pycuda_plus.core.context import CudaContextManager
from pycuda_plus.utils.prebuilt_kernels import get_prebuilt_kernels, get_kernel_names  # Import kernels and names

def test_prebuilt_kernels():
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

            if kernel_name == "vector_add":
                # Test vector addition
                N = 1024
                A = np.random.rand(N).astype(np.float32)
                B = np.random.rand(N).astype(np.float32)
                C = np.zeros_like(A)

                d_A = memory_manager.allocate_device_array(A.shape, dtype=np.float32)
                d_B = memory_manager.allocate_device_array(B.shape, dtype=np.float32)
                d_C = memory_manager.allocate_device_array(C.shape, dtype=np.float32)

                memory_manager.copy_to_device(A, d_A)
                memory_manager.copy_to_device(B, d_B)

                block_size = 256
                grid_size = (N + block_size - 1) // block_size

                kernel_func = kernel.get_kernel(kernel_name)
                kernel.launch_kernel(kernel_func, (grid_size, 1, 1), (block_size, 1, 1), d_A, d_B, d_C, np.int32(N))

                memory_manager.copy_to_host(d_C, C)
                np.testing.assert_allclose(C, A + B, rtol=1e-5)
                print("Vector addition passed.")

            elif kernel_name == "vector_scale":
                # Test vector scaling
                N = 1024
                A = np.random.rand(N).astype(np.float32)
                scalar = 2.5
                B = np.zeros_like(A)

                d_A = memory_manager.allocate_device_array(A.shape, dtype=np.float32)
                d_B = memory_manager.allocate_device_array(B.shape, dtype=np.float32)

                memory_manager.copy_to_device(A, d_A)

                block_size = 256
                grid_size = (N + block_size - 1) // block_size

                kernel_func = kernel.get_kernel(kernel_name)
                kernel.launch_kernel(kernel_func, (grid_size, 1, 1), (block_size, 1, 1), d_A, d_B, np.float32(scalar), np.int32(N))

                memory_manager.copy_to_host(d_B, B)
                np.testing.assert_allclose(B, A * scalar, rtol=1e-5)
                print("Vector scaling passed.")

            elif kernel_name == "matrix_multiply":
                # Test matrix multiplication
                M, N, K = 32, 32, 32
                A = np.random.rand(M, N).astype(np.float32)
                B = np.random.rand(N, K).astype(np.float32)
                C = np.zeros((M, K), dtype=np.float32)

                d_A = memory_manager.allocate_device_array(A.shape, dtype=np.float32)
                d_B = memory_manager.allocate_device_array(B.shape, dtype=np.float32)
                d_C = memory_manager.allocate_device_array(C.shape, dtype=np.float32)

                memory_manager.copy_to_device(A, d_A)
                memory_manager.copy_to_device(B, d_B)

                block_size = (16, 16, 1)
                grid_size = ((K + block_size[0] - 1) // block_size[0], (M + block_size[1] - 1) // block_size[1], 1)

                kernel_func = kernel.get_kernel(kernel_name)
                kernel.launch_kernel(kernel_func, grid_size, block_size, d_A, d_B, d_C, np.int32(M), np.int32(N), np.int32(K))

                memory_manager.copy_to_host(d_C, C)
                np.testing.assert_allclose(C, A @ B, rtol=1e-5)
                print("Matrix multiplication passed.")

            elif kernel_name == "element_wise_sigmoid":
                # Test element-wise sigmoid
                N = 1024
                A = np.random.rand(N).astype(np.float32)
                B = np.zeros_like(A)

                d_A = memory_manager.allocate_device_array(A.shape, dtype=np.float32)
                d_B = memory_manager.allocate_device_array(B.shape, dtype=np.float32)

                memory_manager.copy_to_device(A, d_A)

                block_size = 256
                grid_size = (N + block_size - 1) // block_size

                kernel_func = kernel.get_kernel(kernel_name)
                kernel.launch_kernel(kernel_func, (grid_size, 1, 1), (block_size, 1, 1), d_A, d_B, np.int32(N))

                memory_manager.copy_to_host(d_B, B)
                np.testing.assert_allclose(B, 1.0 / (1.0 + np.exp(-A)), rtol=1e-5)
                print("Element-wise sigmoid passed.")

            elif kernel_name == "array_reduction_sum":
                # Test array reduction sum
                N = 1024
                A = np.random.rand(N).astype(np.float32)
                output_size = (N + 256 - 1) // 256
                partial_sums = np.zeros(output_size, dtype=np.float32)

                d_A = memory_manager.allocate_device_array(A.shape, dtype=np.float32)
                d_partial_sums = memory_manager.allocate_device_array(partial_sums.shape, dtype=np.float32)

                memory_manager.copy_to_device(A, d_A)

                block_size = 256
                grid_size = (N + block_size - 1) // block_size

                kernel_func = kernel.get_kernel(kernel_name)
                kernel.launch_kernel(kernel_func, (grid_size, 1, 1), (block_size, 1, 1), d_A, d_partial_sums, np.int32(N))

                memory_manager.copy_to_host(d_partial_sums, partial_sums)
                total_sum = np.sum(partial_sums)
                np.testing.assert_allclose(total_sum, np.sum(A), rtol=1e-5)
                print("Array reduction sum passed.")

            elif kernel_name == "parallel_max":
                # Test parallel max
                N = 1024
                A = np.random.rand(N).astype(np.float32)
                output_size = (N + 256 - 1) // 256
                partial_maxes = np.full(output_size, -np.inf, dtype=np.float32)

                d_A = memory_manager.allocate_device_array(A.shape, dtype=np.float32)
                d_partial_maxes = memory_manager.allocate_device_array(partial_maxes.shape, dtype=np.float32)

                memory_manager.copy_to_device(A, d_A)

                block_size = 256
                grid_size = (N + block_size - 1) // block_size

                kernel_func = kernel.get_kernel(kernel_name)
                kernel.launch_kernel(kernel_func, (grid_size, 1, 1), (block_size, 1, 1), d_A, d_partial_maxes, np.int32(N))

                memory_manager.copy_to_host(d_partial_maxes, partial_maxes)
                max_value = np.max(partial_maxes)
                np.testing.assert_allclose(max_value, np.max(A), rtol=1e-5)
                print("Parallel max passed.")

            elif kernel_name == "normalize_vector":
                # Test normalize vector
                N = 1024
                A = np.random.rand(N).astype(np.float32)
                norm = np.linalg.norm(A)
                B = np.zeros_like(A)

                d_A = memory_manager.allocate_device_array(A.shape, dtype=np.float32)
                d_B = memory_manager.allocate_device_array(B.shape, dtype=np.float32)

                memory_manager.copy_to_device(A, d_A)

                block_size = 256
                grid_size = (N + block_size - 1) // block_size

                kernel_func = kernel.get_kernel(kernel_name)
                kernel.launch_kernel(kernel_func, (grid_size, 1, 1), (block_size, 1, 1), d_A, d_B, np.float32(norm), np.int32(N))

                memory_manager.copy_to_host(d_B, B)
                np.testing.assert_allclose(B, A / norm, rtol=1e-5)
                print("Normalize vector passed.")

            elif kernel_name == "relu_activation":
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
            
            elif kernel_name == "tanh_activation":
                # Test Tanh Activation
                N = 1024
                # Random input array
                inputs = np.random.randn(N).astype(np.float32)
                outputs = np.zeros_like(inputs)

                # Allocate device memory
                d_inputs = memory_manager.allocate_device_array(inputs.shape, dtype=np.float32)
                d_outputs = memory_manager.allocate_device_array(outputs.shape, dtype=np.float32)

                memory_manager.copy_to_device(inputs, d_inputs)

                block_size = 256
                grid_size = (N + block_size - 1) // block_size

                kernel_func = kernel.get_kernel('tanh_activation')
                kernel.launch_kernel(kernel_func, (grid_size, 1, 1), (block_size, 1, 1), 
                                    d_inputs, d_outputs, np.int32(N))

                # Copy the results back to host
                memory_manager.copy_to_host(d_outputs, outputs)

                # Compute expected output manually using NumPy
                expected_outputs = np.tanh(inputs)

                # Debug print results
                print(f"Computed outputs: {outputs[:10]}")  # Print a subset for verification
                print(f"Expected outputs: {expected_outputs[:10]}")  # Adjust the debug output as necessary
                np.testing.assert_allclose(outputs, expected_outputs, rtol=1e-5, atol=1e-5)

                print("Tanh activation passed.")

            elif kernel_name == "adam_optimizer":
                # Test Adam Optimizer
                N = 1024
                # Initialize weights, gradients, and momentums
                weights = np.random.randn(N).astype(np.float32)
                gradients = np.random.randn(N).astype(np.float32)
                m = np.zeros_like(weights, dtype=np.float32)
                v = np.zeros_like(weights, dtype=np.float32)
                beta1 = 0.9
                beta2 = 0.999
                epsilon = 1e-8
                lr = 0.001
                t = 1  # iteration count

                # Allocate device memory
                d_weights = memory_manager.allocate_device_array(weights.shape, dtype=np.float32)
                d_gradients = memory_manager.allocate_device_array(gradients.shape, dtype=np.float32)
                d_m = memory_manager.allocate_device_array(m.shape, dtype=np.float32)
                d_v = memory_manager.allocate_device_array(v.shape, dtype=np.float32)
                
                memory_manager.copy_to_device(weights, d_weights)
                memory_manager.copy_to_device(gradients, d_gradients)
                memory_manager.copy_to_device(m, d_m)
                memory_manager.copy_to_device(v, d_v)

                block_size = 256
                grid_size = (N + block_size - 1) // block_size

                kernel_func = kernel.get_kernel('adam_optimizer')
                kernel.launch_kernel(kernel_func, (grid_size, 1, 1), (block_size, 1, 1), 
                                    d_weights, d_gradients, d_m, d_v, np.float32(beta1), 
                                    np.float32(beta2), np.float32(epsilon), np.float32(lr), np.int32(t), np.int32(N))

                # Copy the results back to host
                memory_manager.copy_to_host(d_weights, weights)
                memory_manager.copy_to_host(d_m, m)
                memory_manager.copy_to_host(d_v, v)

                # Now compute the expected update manually using the Adam formula
                m_correct = beta1 * m + (1 - beta1) * gradients
                v_correct = beta2 * v + (1 - beta2) * gradients ** 2
                m_hat = m_correct / (1 - beta1 ** t)
                v_hat = v_correct / (1 - beta2 ** t)
                weights_updated = weights - lr * m_hat / (np.sqrt(v_hat) + epsilon)

                # Debug print results
                print(f"Updated weights: {weights_updated[:10]}")  # Print a subset for verification
                print(f"Expected updated weights: {weights[:10]}")  # Adjust the debug output as necessary
                np.testing.assert_allclose(weights, weights_updated, rtol=1e-3, atol=1e-4)

                print("Adam optimizer passed.")

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
    finally:
        context_manager.finalize_context()
        print("All prebuilt kernel tests passed.")

if __name__ == "__main__":
    test_prebuilt_kernels()
