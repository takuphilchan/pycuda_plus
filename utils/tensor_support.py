"""
Utilities for integrating PyTorch and TensorFlow tensors with PyCUDA.
"""
import torch
import tensorflow as tf
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

def torch_to_device(tensor):
    """Convert a PyTorch tensor to a PyCUDA device array."""
    numpy_array = tensor.cpu().numpy()
    return gpuarray.to_gpu(numpy_array)

def device_to_torch(device_array, dtype=torch.float):
    """Convert a PyCUDA device array to a PyTorch tensor."""
    numpy_array = device_array.get()
    return torch.tensor(numpy_array, dtype=dtype)

def tf_to_device(tensor):
    """Convert a TensorFlow tensor to a PyCUDA device array."""
    numpy_array = tensor.numpy()
    return gpuarray.to_gpu(numpy_array)

def device_to_tf(device_array, dtype=tf.float32):
    """Convert a PyCUDA device array to a TensorFlow tensor."""
    numpy_array = device_array.get()
    return tf.convert_to_tensor(numpy_array, dtype=dtype)
