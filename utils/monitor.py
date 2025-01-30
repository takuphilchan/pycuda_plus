import time
import threading
from pycuda_plus.core.memory import MemoryManager
from pycuda_plus.core.context import CudaContextManager

# Global shared memory data to be updated and used by Dash
memory_data = {
    'timestamps': [],
    'used_memory': [],
    'free_memory': []
}

# Lock for thread-safe access to memory_data and kernel_data
data_lock = threading.Lock()

def get_memory_usage():
    """
    Get memory data (used, free) for real-time updates.
    """
    memory_manager = MemoryManager()
    total_memory, used_memory, free_memory = memory_manager.get_memory_info()
    return total_memory, used_memory, free_memory

def start_memory_monitoring():
    """
    Starts real-time memory monitoring and stores data for Dash updates.
    """
    def monitor_memory():
        # Initialize the CUDA context using the context manager
        context_manager = CudaContextManager()
        context_manager.initialize_context()
        
        try:
            while True:
                try:
                    total_memory, used_memory, free_memory = get_memory_usage()
                    print(used_memory, free_memory)
                    with data_lock:  # Ensure thread-safe updates to memory_data
                        memory_data['timestamps'].append(len(memory_data['timestamps']))
                        memory_data['used_memory'].append(used_memory)
                        memory_data['free_memory'].append(free_memory)
                except Exception as e:
                    print(f"Error retrieving memory info: {e}")
                    with data_lock:
                        memory_data['timestamps'].append(len(memory_data['timestamps']))
                        memory_data['used_memory'].append(0)
                        memory_data['free_memory'].append(0)
                
                time.sleep(1)  # Update memory data every second
        finally:
            # Finalize the context to ensure proper cleanup
            context_manager.finalize_context()
    
    # Start the memory monitoring in a background thread
    monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
    monitor_thread.start()