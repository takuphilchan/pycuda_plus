import time 
import traceback
import pycuda.driver as cuda
import pycuda.autoinit

class PerformanceProfiler:
    """Provides performance profiling tools for CUDA kernel execution. 🚀🔧"""
    
    def __init__(self):
        self.execution_times = {}
        print("✅ Performance Profiler Initialized.")

    def profile_kernel(self, kernel, grid, block, *args):
        """Profiles the execution time of a CUDA kernel. ⏱️"""
        try:
            if kernel is None:
                raise ValueError("❌ Kernel function is None. Ensure compilation was successful. 🚫")

            if not cuda.Context.get_current():
                raise RuntimeError("❌ CUDA context is inactive. Initialize it before profiling. 🛑")

            print("🔍 Profiling started...")

            start, end = cuda.Event(), cuda.Event()
            start.record()
            
            # Launch the kernel correctly
            kernel(*args, block=block, grid=grid)
            
            end.record()
            end.synchronize()

            exec_time = start.time_till(end) * 1e-3  # Convert ms to seconds
            print(f"⏳ Kernel execution time: {exec_time:.6f} seconds.")

            # Store the execution time (optional: can uncomment below to track)
            # self.execution_times[kernel.__name__] = exec_time
            
            return exec_time
            
        except cuda.LogicError as e:
            print(f"❌ CUDA LogicError: {e} 💥")
            traceback.print_exc()
        except RuntimeError as e:
            print(f"❌ CUDA Runtime Error: {e} 💥")
            traceback.print_exc()
        except Exception as e:
            print(f"❌ Unexpected error in profiling kernel: {e} 🤔")
            traceback.print_exc()
            
        return None

    def get_kernel_execution_times(self):
        """Returns recorded execution times of kernels. 📊"""
        print("📋 Fetching kernel execution times...")
        return self.execution_times

    def display_execution_times(self):
        """Displays execution times of recorded kernels. 🖥️📈"""
        if not self.execution_times:
            print("⚠️ No kernel execution times recorded yet.")
            return
        print("📝 Kernel Execution Times:")
        for kernel, exec_time in self.execution_times.items():
            print(f"🔹 {kernel}: {exec_time:.6f} seconds")
