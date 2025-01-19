"""
Library of commonly used CUDA kernels.
"""
from pycuda.compiler import SourceModule

kernels = {
    "vector_add": """__global__ void vector_add(float *a, float *b, float *c, int n) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < n) {
            c[idx] = a[idx] + b[idx];
        }
    }""",
}

def get_kernel(kernel_name):
    """Retrieve and compile a prebuilt kernel by name."""
    if kernel_name in kernels:
        module = SourceModule(kernels[kernel_name])
        return module.get_function(kernel_name)
    else:
        raise ValueError(f"Kernel {kernel_name} not found.")
