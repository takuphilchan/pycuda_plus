kernels = {
    # General kernels
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
    
    "element_wise_sigmoid": """__global__ void element_wise_sigmoid(float *a, float *b, int n) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < n) {
            b[idx] = 1.0f / (1.0f + expf(-a[idx]));
        }
    }""",
    
    "array_reduction_sum": """__global__ void array_reduction_sum(float *input, float *output, int n) {
        __shared__ float sdata[256];
        
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        
        sdata[tid] = (i < n) ? input[i] : 0;
        __syncthreads();
        
        for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
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
        
        for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
            }
            __syncthreads();
        }
        
        if (tid == 0) output[blockIdx.x] = sdata[0];
    }""",
    
    "normalize_vector": """__global__ void normalize_vector(float *input, float *output, float norm, int n) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < n) {
            output[idx] = input[idx] / norm;
        }
    }""",
    
    # New Machine Learning and Deep Learning Kernels
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
    
    "dropout": """__global__ void dropout(float *input, float *output, float *random_values, 
                          float dropout_rate, int n) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < n) {
            if (random_values[idx] < dropout_rate) {
                output[idx] = 0.0f;
            } else {
                output[idx] = input[idx] / (1.0f - dropout_rate);
            }
        }
    }"""
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