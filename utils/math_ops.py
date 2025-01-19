"""
High-level mathematical operations for PyCUDA.
"""
from pycuda.elementwise import ElementwiseKernel

# Predefined elementwise kernels for common operations
add_kernel = ElementwiseKernel(
    "float *a, float *b, float *c",
    "c[i] = a[i] + b[i];",
    "add_kernel"
)

def vector_add(a, b, c):
    """Perform elementwise addition of two vectors on the GPU."""
    add_kernel(a, b, c)