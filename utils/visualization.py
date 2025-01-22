import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import seaborn as sns
import pandas as pd

def plot_memory_usage(total_memory: int, used_memory: int, colors: List[str] = ['#ff9999', '#66b3ff']):
    """
    Visualize GPU memory usage as a pie chart with enhanced aesthetics.
    
    Parameters:
        total_memory (int): Total GPU memory in bytes.
        used_memory (int): Used GPU memory in bytes.
        colors (List[str]): Colors for the pie chart segments.
    """
    if total_memory <= 0 or used_memory < 0:
        raise ValueError("Total memory must be positive, and used memory must be non-negative.")
    
    free_memory = total_memory - used_memory
    if free_memory < 0:
        raise ValueError("Used memory cannot exceed total memory.")
    
    labels = ['Used Memory', 'Free Memory']
    sizes = [used_memory, free_memory]
    
    plt.figure(figsize=(7, 7))
    wedges, texts, autotexts = plt.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'black', 'linewidth': 1.2}
    )
    
    # Add shadows to create depth
    for wedge in wedges:
        wedge.set_edgecolor('black')
        wedge.set_linewidth(1.5)
    
    # Enhance the text on the pie chart
    for autotext in autotexts:
        autotext.set_fontsize(14)
        autotext.set(weight='bold')
    
    plt.axis('equal')
    plt.title('GPU Memory Usage', fontsize=18, weight='bold')
    plt.show()

def plot_execution_times(execution_times):
    """
    Visualize execution times for multiple kernels or operations with improved aesthetics.

    Parameters:
        execution_times (dict): Dictionary with kernel/operation names as keys and lists of execution times (in ms) as values.
    """
    sns.set_theme(style="whitegrid", context="talk")

    # Prepare data for plotting
    data = []
    for kernel, times in execution_times.items():
        for time in times:
            data.append({'Kernel': kernel, 'Execution Time (ms)': time})

    df = pd.DataFrame(data)

    # Aggregate data for error bars (mean and standard deviation)
    aggregated_data = (
        df.groupby('Kernel')['Execution Time (ms)']
        .agg(['mean', 'std'])
        .reset_index()
        .rename(columns={'mean': 'Mean Time', 'std': 'Std Dev'})
    )

    # Plot with error bars
    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=aggregated_data,
        x='Kernel',
        y='Mean Time',
        palette='muted',  # A more muted color palette for a professional look
        errorbar=None,  # Explicitly disable error bars (no need for `ci=None`)
    )

    # Add error bars manually
    for i, row in aggregated_data.iterrows():
        plt.errorbar(
            x=i, 
            y=row['Mean Time'], 
            yerr=row['Std Dev'], 
            fmt='none', 
            ecolor='black', 
            capsize=5, 
            alpha=0.8
        )

    # Add labels and title
    plt.title('Kernel Execution Times with Error Bars', fontsize=20, weight='bold', pad=20)
    plt.xlabel('Kernel/Operation', fontsize=16, weight='bold')
    plt.ylabel('Mean Execution Time (ms)', fontsize=16, weight='bold')
    plt.xticks(fontsize=14, rotation=0, ha='center')
    plt.yticks(fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add data labels above bars
    for i, row in aggregated_data.iterrows():
        plt.text(
            x=i, 
            y=row['Mean Time'] + row['Std Dev'] + 0.5, 
            s=f"{row['Mean Time']:.2f} ms", 
            ha='center', 
            fontsize=12, 
            color='black', 
            fontweight='bold'
        )

    plt.tight_layout()
    plt.show()

def real_time_memory_monitor(get_memory_usage_func, duration: int = 10, interval: float = 0.5):
    """
    Monitor GPU memory usage in real-time and plot dynamically with improved aesthetics.
    
    Parameters:
        get_memory_usage_func (function): Function to retrieve current memory usage as (total_memory, used_memory).
        duration (int): Total monitoring duration in seconds.
        interval (float): Update interval in seconds.
    """
    import time
    
    if duration <= 0 or interval <= 0:
        raise ValueError("Duration and interval must be positive.")
    
    total_memory, used_memory = get_memory_usage_func()
    timestamps = []
    used_memories = []
    free_memories = []
    
    start_time = time.time()
    while time.time() - start_time < duration:
        total_memory, used_memory = get_memory_usage_func()
        free_memory = total_memory - used_memory
        
        timestamps.append(time.time() - start_time)
        used_memories.append(used_memory)
        free_memories.append(free_memory)
        
        plt.clf()
        plt.plot(timestamps, used_memories, label='Used Memory', color='#FF6347', linestyle='-', marker='o', markersize=6)
        plt.plot(timestamps, free_memories, label='Free Memory', color='#4682B4', linestyle='-', marker='s', markersize=6)
        plt.xlabel('Time (s)', fontsize=14, weight='bold')
        plt.ylabel('Memory (Bytes)', fontsize=14, weight='bold')
        plt.title('Real-Time GPU Memory Usage', fontsize=16, weight='bold')
        plt.legend(loc='upper right', fontsize=12)
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.pause(interval)
    
    plt.tight_layout()
    plt.show()

