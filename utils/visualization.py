"""
Visualization tools for PyCUDA data and performance.
"""
import matplotlib.pyplot as plt

def plot_memory_usage(total_memory, used_memory):
    """Visualize GPU memory usage."""
    labels = ['Used Memory', 'Free Memory']
    sizes = [used_memory, total_memory - used_memory]
    colors = ['#ff9999','#66b3ff']
    
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('GPU Memory Usage')
    plt.show()