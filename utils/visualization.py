import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import seaborn as sns
import pandas as pd

# Define a consistent color palette with slightly more contrast
COLORS = {
    'primary': '#3498db',      # Slightly more saturated soft blue
    'secondary': '#e74c3c',    # Original soft red moved to secondary
    'accent': '#2c3e50',       # Dark slate (unchanged)
    'highlight': '#27ae60',    # Soft green for additional contrast
    'background': '#ecf0f1'    # Light gray (unchanged)
}

def plot_memory_usage(total_memory: int, used_memory: int, colors: List[str] = None):
    """
    Visualize GPU memory usage as a pie chart with enhanced aesthetics.
    """
    if colors is None:
        colors = [COLORS['secondary'], COLORS['primary']]  # Red for used, blue for free
        
    if total_memory <= 0 or used_memory < 0:
        raise ValueError("Total memory must be positive, and used memory must be non-negative.")
    
    free_memory = total_memory - used_memory
    if free_memory < 0:
        raise ValueError("Used memory cannot exceed total memory.")
    
    labels = ['Used Memory', 'Free Memory']
    sizes = [used_memory, free_memory]
    
    plt.figure(figsize=(7, 7), facecolor='white')
    wedges, texts, autotexts = plt.pie(
        sizes, 
        labels=labels, 
        colors=colors, 
        autopct='%1.1f%%', 
        startangle=90, 
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    
    # Enhance text appearance
    plt.setp(autotexts, color=COLORS['accent'], fontsize=12, weight='bold')
    plt.setp(texts, color=COLORS['accent'], fontsize=12, weight='bold')
    
    plt.axis('equal')
    plt.title('GPU Memory Usage', fontsize=16, weight='bold', color=COLORS['accent'], pad=20)
    plt.show()

def real_time_memory_monitor(get_memory_usage_func, duration: int = 10, interval: float = 0.5):
    """
    Monitor GPU memory usage in real-time with improved aesthetics.
    """
    import time
    
    if duration <= 0 or interval <= 0:
        raise ValueError("Duration and interval must be positive.")
    
    plt.style.use('seaborn')
    
    # Create figure once, outside the loop
    fig = plt.figure(figsize=(8, 6), facecolor='white')
    
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
        
        # Plot with more contrasting colors
        plt.plot(timestamps, used_memories, 
                label='Used Memory', 
                color=COLORS['secondary'],  # Red for used memory
                linestyle='-', 
                marker='o', 
                markersize=6,
                markeredgecolor='white',
                markeredgewidth=1,
                linewidth=2)
        
        plt.plot(timestamps, free_memories, 
                label='Free Memory', 
                color=COLORS['primary'],  # Blue for free memory
                linestyle='-', 
                marker='s', 
                markersize=6,
                markeredgecolor='white',
                markeredgewidth=1,
                linewidth=2)
        
        plt.xlabel('Time (s)', fontsize=12, weight='bold', color=COLORS['accent'])
        plt.ylabel('Memory (Bytes)', fontsize=12, weight='bold', color=COLORS['accent'])
        plt.title('Real-Time GPU Memory Usage', fontsize=14, weight='bold', color=COLORS['accent'])
        
        # Enhanced legend
        legend = plt.legend(loc='upper right', fontsize=10, framealpha=0.9,
                          edgecolor=COLORS['accent'])
        legend.get_frame().set_facecolor('white')
        
        plt.grid(True, which='both', linestyle='--', alpha=0.3, color=COLORS['accent'])
        plt.tick_params(colors=COLORS['accent'])
        
        plt.tight_layout()
        plt.pause(interval)
    
    plt.show()

def plot_execution_times(execution_times):
    """
    Visualize execution times for multiple kernels or operations with improved aesthetics.
    """
    # Set custom style to match real-time plot
    plt.style.use('seaborn')  # This gives us the background grid
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = COLORS['background']  # Light gray background

    data = []
    for kernel, times in execution_times.items():
        for time in times:
            data.append({'Kernel': kernel, 'Execution Time (ms)': time})

    df = pd.DataFrame(data)
    aggregated_data = (
        df.groupby('Kernel')['Execution Time (ms)']
        .agg(['mean', 'std'])
        .reset_index()
        .rename(columns={'mean': 'Mean Time', 'std': 'Std Dev'})
    )

    plt.figure(figsize=(8, 7))

    # Create custom palette using our colors
    base_colors = [
        COLORS['primary'],      # Soft blue
        COLORS['secondary'],    # Soft red
        COLORS['highlight'],    # Soft green
    ]
    
    # Generate additional shades if needed
    n_colors = len(aggregated_data)
    if n_colors > len(base_colors):
        custom_palette = sns.color_palette([COLORS['primary']], n_colors=n_colors)
    else:
        custom_palette = base_colors[:n_colors]

    # Create the background style
    ax = plt.gca()
    ax.set_facecolor(COLORS['background'])  # Set background color
    ax.spines['top'].set_visible(False)     # Remove top border
    ax.spines['right'].set_visible(False)   # Remove right border
    
    sns.barplot(
        data=aggregated_data,
        x='Kernel',
        y='Mean Time',
        hue='Kernel',
        legend=False,
        palette=custom_palette,
        errorbar=None,
    )

    # Add error bars
    for i, row in aggregated_data.iterrows():
        plt.errorbar(
            x=i, 
            y=row['Mean Time'], 
            yerr=row['Std Dev'], 
            fmt='none', 
            ecolor=COLORS['accent'], 
            capsize=5, 
            alpha=0.5
        )

    plt.title('Kernel Execution Times', fontsize=16, weight='bold', color=COLORS['accent'], pad=20)
    plt.xlabel('Kernel/Operation', fontsize=12, weight='bold', color=COLORS['accent'], labelpad=10)
    plt.ylabel('Mean Execution Time (ms)', fontsize=12, weight='bold', color=COLORS['accent'], labelpad=10)
    
    plt.xticks(fontsize=10, rotation=0, ha='center', color=COLORS['accent'])
    plt.yticks(fontsize=10, color=COLORS['accent'])
    
    # Enhance grid style: Show only horizontal grid lines
    plt.grid(True, axis='y', linestyle='--', alpha=0.3, color=COLORS['accent'])

    # Add data labels
    for i, row in aggregated_data.iterrows():
        value = row['Mean Time']
        if value < 0.01:
            format_str = f"{value:.6f}"
        elif value < 0.1:
            format_str = f"{value:.4f}"
        else:
            format_str = f"{value:.3f}"
            
        plt.text(
            x=i, 
            y=row['Mean Time'] + row['Std Dev'] + (row['Mean Time'] * 0.02),
            s=f"{format_str} ms", 
            ha='center', 
            fontsize=9,
            color=COLORS['accent'],
            weight='bold'
        )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()
