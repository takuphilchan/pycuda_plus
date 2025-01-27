kernels = {
    # General Kernels
    "vector_add": """__global__ void vector_add(float *a, float *b, float *c, int n) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < n) {
            c[idx] = a[idx] + b[idx];
        }
    }""",
    
    "vector_scale": """__global__ void vector_scale(float *a, float *b, float scalar, int n) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < n) {
            b[idx] = a[idx] * scalar;
        }
    }""",
    
    "matrix_multiply": """__global__ void matrix_multiply(float *a, float *b, float *c, int M, int N, int K) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (row < M && col < K) {
            float sum = 0.0f;
            for (int i = 0; i < N; i++) {
                sum += a[row * N + i] * b[i * K + col];
            }
            c[row * K + col] = sum;
        }
    }""",
    
    "matrix_transpose": """__global__ void matrix_transpose(float *input, float *output, int rows, int cols) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x < cols && y < rows) {
            output[x * rows + y] = input[y * cols + x];
        }
    }""",

    "dot_product": """__global__ void dot_product(float *a, float *b, float *result, int n) {
        __shared__ float partial_sum[256];
        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        partial_sum[tid] = (idx < n) ? a[idx] * b[idx] : 0.0f;
        __syncthreads();
        
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                partial_sum[tid] += partial_sum[tid + s];
            }
            __syncthreads();
        }
        if (tid == 0) atomicAdd(result, partial_sum[0]);
    }""",
    
    # Reduction Kernels
    "array_reduction_sum": """__global__ void array_reduction_sum(float *input, float *output, int n) {
        __shared__ float sdata[256];
        
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        
        sdata[tid] = (i < n) ? input[i] : 0;
        __syncthreads();
        
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        
        if (tid == 0) output[blockIdx.x] = sdata[0];
    }""",
    
    "parallel_max": """__global__ void parallel_max(float *input, float *output, int n) {
        __shared__ float sdata[256];
        
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        
        sdata[tid] = (i < n) ? input[i] : -INFINITY;
        __syncthreads();
        
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
            }
            __syncthreads();
        }
        
        if (tid == 0) output[blockIdx.x] = sdata[0];
    }""",

    # Machine Learning Kernels
    "batch_normalization": """__global__ void batch_normalization(float *input, float *output, 
                                            float *mean, float *variance,
                                            float *gamma, float *beta,
                                            float epsilon, int n) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < n) {
            output[idx] = gamma[0] * (input[idx] - mean[0]) / sqrtf(variance[0] + epsilon) + beta[0];
        }
    }""",
    # Deep Learning Kernels
    "convolution_2d": """__global__ void convolution_2d(float *input, float *kernel, float *output, 
                                  int input_width, int input_height, 
                                  int kernel_width, int kernel_height) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (x < input_width - kernel_width + 1 && y < input_height - kernel_height + 1) {
            float sum = 0.0f;
            for (int i = 0; i < kernel_height; ++i) {
                for (int j = 0; j < kernel_width; ++j) {
                    sum += input[(y + i) * input_width + (x + j)] * kernel[i * kernel_width + j];
                }
            }
            output[y * (input_width - kernel_width + 1) + x] = sum;
        }
    }""",
    
    "max_pooling_2d": """__global__ void max_pooling_2d(float *input, float *output, int input_width, int input_height, 
                               int pool_width, int pool_height, int stride) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (x < (input_width - pool_width) / stride + 1 && y < (input_height - pool_height) / stride + 1) {
            float max_val = -INFINITY;
            for (int i = 0; i < pool_height; ++i) {
                for (int j = 0; j < pool_width; ++j) {
                    max_val = fmaxf(max_val, input[(y * stride + i) * input_width + (x * stride + j)]);
                }
            }
            output[y * ((input_width - pool_width) / stride + 1) + x] = max_val;
        }
    }""",

    # Activation Functions
    "relu_activation": """__global__ void relu_activation(float *input, float *output, int n) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < n) {
            output[idx] = fmaxf(0.0f, input[idx]);
        }
    }""",
    
    "leaky_relu_activation": """__global__ void leaky_relu_activation(float *input, float *output, float alpha, int n) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < n) {
            output[idx] = input[idx] > 0 ? input[idx] : alpha * input[idx];
        }
    }""",
    
    "tanh_activation": """__global__ void tanh_activation(float *input, float *output, int n) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < n) {
            output[idx] = tanhf(input[idx]);
        }
    }""",
    "softmax_activation": """__global__ void softmax_activation(float *input, float *output, int n) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < n) {
            output[idx] = expf(input[idx]);
        }
    }""",
    # Optimizer
    "adam_optimizer": """__global__ void adam_optimizer(float *weights, float *gradients, float *m, float *v, 
                                     float learning_rate, float beta1, float beta2, 
                                     float epsilon, int t, int n) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < n) {
            // Update biased first moment estimate
            m[idx] = beta1 * m[idx] + (1.0f - beta1) * gradients[idx];
            // Update biased second raw moment estimate
            v[idx] = beta2 * v[idx] + (1.0f - beta2) * gradients[idx] * gradients[idx];
            
            // Compute bias-corrected first moment estimate
            float m_hat = m[idx] / (1.0f - powf(beta1, t));
            // Compute bias-corrected second raw moment estimate
            float v_hat = v[idx] / (1.0f - powf(beta2, t));
            
            // Update weights
            weights[idx] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
        }
    }""",
    
    # Image Processing Kernels
    "rgb_to_grayscale": """__global__ void rgb_to_grayscale(float *d_input, float *d_output, int height, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        float r = d_input[3 * idx];
        float g = d_input[3 * idx + 1];
        float b = d_input[3 * idx + 2];
        d_output[idx] = 0.2989f * r + 0.5870f * g + 0.1140f * b;
        }
    }
    """,
        
    "gaussian_blur": """__global__ void gaussian_blur(float *input, float *output, int width, int height, float *kernel, int kernel_size) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (x < width && y < height) {
            float sum = 0.0f;
            int half_k = kernel_size / 2;
            for (int i = -half_k; i <= half_k; ++i) {
                for (int j = -half_k; j <= half_k; ++j) {
                    int xi = min(max(x + i, 0), width - 1);
                    int yi = min(max(y + j, 0), height - 1);
                    sum += input[yi * width + xi] * kernel[(i + half_k) * kernel_size + (j + half_k)];
                }
            }
            output[y * width + x] = sum;
        }
    }""",
}

def get_prebuilt_kernels():
    """
    Return the dictionary of prebuilt kernels.
    
    Returns:
        dict: Prebuilt CUDA kernel source codes
    """
    return kernels


def get_kernel_names():
    """
    Return list of available prebuilt kernel names.
    
    Returns:
        list: Names of prebuilt kernels
    """
    return list(kernels.keys())