import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_heatmap(json_file, metric='qbadic', base=None, multiplier=None, generation='final'):
    """Create a heatmap showing correlation by base and multiplier for a specific metric.
    
    Args:
        json_file: Path to the metrics JSON file
        metric: Which distance metric to use ('qbadic', 'euclidean', 'padic', etc.)
        base: For qbadic, specific base to visualize (if None, creates matrix of all bases)
        multiplier: For qbadic, specific multiplier to visualize (if None, creates matrix of all multipliers)
        generation: 'final', 'initial', 'average', or specific generation number
    """
    # Load data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    metrics_tracked = data['metrics_tracked']
    metrics_data = data['metrics']
    
    # Validate metric
    if metric not in metrics_tracked:
        raise ValueError(f"Metric '{metric}' not found. Available metrics: {metrics_tracked}")
    
    # Determine which generation to use
    if generation == 'average':
        gen_label = "Promedio de todas las generaciones"
        use_average = True
    elif generation == 'final':
        gen_idx = -1
        gen_label = f"Generacion {metrics_data['generation'][-1]}"
        use_average = False
    elif generation == 'initial':
        gen_idx = 0
        gen_label = f"Generacion {metrics_data['generation'][0]}"
        use_average = False
    else:
        gen_idx = metrics_data['generation'].index(generation)
        gen_label = f"Generacion {generation}"
        use_average = False
    
    # Handle qbadic metric
    if metric == 'qbadic':
        bases = data['qbadic_bases']
        multipliers = data['multipliers']
        
        # Build correlation matrix
        correlation_matrix = np.zeros((len(bases), len(multipliers)))
        
        for i, b in enumerate(bases):
            for j, m in enumerate(multipliers):
                key = f'qbadic_b{b}_mult{m}_correlation'
                if use_average:
                    # Filter out None values before averaging
                    values = [v for v in metrics_data[key] if v is not None]
                    correlation_matrix[i, j] = np.mean(values) if values else np.nan
                else:
                    value = metrics_data[key][gen_idx]
                    correlation_matrix[i, j] = value if value is not None else np.nan
        
        x_labels = multipliers
        y_labels = bases
        xlabel = 'Multiplicador'
        ylabel = 'Base'
        title_metric = 'qb-ádico'
        
    else:
        # Handle non-qbadic metrics (euclidean, padic, etc.)
        key = f'{metric.replace("-", "")}_correlation'
        
        if key not in metrics_data:
            raise ValueError(f"Correlation data for metric '{metric}' not found in the JSON file")
        
        if use_average:
            values = [v for v in metrics_data[key] if v is not None]
            corr_value = np.mean(values) if values else np.nan
        else:
            value = metrics_data[key][gen_idx]
            corr_value = value if value is not None else np.nan
        
        # For single-value metrics, create a 1x1 matrix
        correlation_matrix = np.array([[corr_value]])
        x_labels = [metric.capitalize()]
        y_labels = ['']
        xlabel = 'Métrica'
        ylabel = ''
        title_metric = metric.capitalize()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Mask NaN values
    mask = np.isnan(correlation_matrix)
    
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True if correlation_matrix.size <= 20 else True,  # Show values if matrix is small
                fmt='.3f',
                cmap='RdYlGn',  # Red-Yellow-Green for correlation
                center=0,  # Center the colormap at 0
                vmin=-0.25,   # Correlation ranges from -1 to 1
                vmax=0.25,
                xticklabels=x_labels,
                yticklabels=y_labels,
                cbar_kws={'label': 'Correlación'},
                ax=ax)
    
    if len(y_labels) > 1:
        ax.invert_yaxis()
    
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(f'Correlación (Diferencia de Fitness vs Distancia {title_metric})\n{gen_label}', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save with appropriate filename
    save_name = 'average' if use_average else str(generation)
    metric_name = metric if metric != 'qbadic' else 'qbadic_matrix'
    plt.savefig(f'correlation_heatmap_{metric_name}_{save_name}.png', dpi=300, bbox_inches='tight')
    print(f"Heatmap saved as 'correlation_heatmap_{metric_name}_{save_name}.png'")
    plt.show()
    
    # Print summary statistics (only for matrices, not single values)
    if correlation_matrix.size > 1:
        valid_values = correlation_matrix[~mask]
        if len(valid_values) > 0:
            print(f"\n{gen_label} Correlation Statistics ({metric}):")
            print(f"  Min correlation: {valid_values.min():.6f}")
            print(f"  Max correlation: {valid_values.max():.6f}")
            print(f"  Mean correlation: {valid_values.mean():.6f}")
            
            if metric == 'qbadic':
                max_pos = np.unravel_index(np.nanargmax(correlation_matrix), correlation_matrix.shape)
                min_pos = np.unravel_index(np.nanargmin(correlation_matrix), correlation_matrix.shape)
                print(f"\nHighest correlation: Base={y_labels[max_pos[0]]}, Multiplier={x_labels[max_pos[1]]}")
                print(f"Lowest correlation: Base={y_labels[min_pos[0]]}, Multiplier={x_labels[min_pos[1]]}")
        else:
            print(f"\n{gen_label}: No valid correlation data available")
    else:
        if not np.isnan(correlation_matrix[0, 0]):
            print(f"\n{gen_label} Correlation ({metric}): {correlation_matrix[0, 0]:.6f}")
        else:
            print(f"\n{gen_label}: No valid correlation data available for {metric}")

def plot_diversity_heatmap(json_file, generation='final'):
    """Create a heatmap showing qb-adic diversity by base and multiplier."""
    # Load data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    bases = data['qbadic_bases']
    multipliers = data['multipliers']
    metrics = data['metrics']
    
    # Determine which generation to use
    if generation == 'average':
        gen_label = "Promedio de todas las generaciones"
        use_average = True
    elif generation == 'final':
        gen_idx = -1
        gen_label = f"Generacion {metrics['generation'][-1]}"
        use_average = False
    elif generation == 'initial':
        gen_idx = 0
        gen_label = f"Generacion {metrics['generation'][0]}"
        use_average = False
    else:
        gen_idx = metrics['generation'].index(generation)
        gen_label = f"Generacion {generation}"
        use_average = False
    
    # Build diversity matrix
    diversity_matrix = np.zeros((len(bases), len(multipliers)))
    
    for i, base in enumerate(bases):
        for j, mult in enumerate(multipliers):
            key = f'qbadic_b{base}_mult{mult}_diversity'
            if use_average:
                diversity_matrix[i, j] = np.mean(metrics[key])
            else:
                diversity_matrix[i, j] = metrics[key][gen_idx]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(diversity_matrix, 
                annot=False, 
                fmt='.2f',
                cmap='YlGnBu',
                xticklabels=multipliers,
                yticklabels=bases,
                cbar_kws={'label': 'Diversidad'},
                ax=ax)
    ax.invert_yaxis()
    
    ax.set_xlabel('Multiplicador', fontsize=12, fontweight='bold')
    ax.set_ylabel('Base', fontsize=12, fontweight='bold')
    ax.set_title(f'Diversidad por base y multiplicador del Funcional de valoración b-ádico\n{gen_label}', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save with appropriate filename
    save_name = 'average' if use_average else str(generation)
    plt.savefig(f'diversity_heatmap_{save_name}.png', dpi=300, bbox_inches='tight')
    print(f"Heatmap saved as 'diversity_heatmap_{save_name}.png'")
    plt.show()
    
    # Print summary statistics
    print(f"\n{gen_label} Diversity Statistics:")
    print(f"  Min diversity: {diversity_matrix.min():.6f}")
    print(f"  Max diversity: {diversity_matrix.max():.6f}")
    print(f"  Mean diversity: {diversity_matrix.mean():.6f}")
    print(f"\nBest (highest diversity): Base={bases[np.unravel_index(diversity_matrix.argmax(), diversity_matrix.shape)[0]]}, "
          f"Multiplier={multipliers[np.unravel_index(diversity_matrix.argmax(), diversity_matrix.shape)[1]]}")
    print(f"Worst (lowest diversity): Base={bases[np.unravel_index(diversity_matrix.argmin(), diversity_matrix.shape)[0]]}, "
          f"Multiplier={multipliers[np.unravel_index(diversity_matrix.argmin(), diversity_matrix.shape)[1]]}")

if __name__ == "__main__":
    json_file = 'metrics/run_3000gen_interval5.json'
    
    # Diversity heatmap (only for qbadic)
    print("Generating diversity heatmap...")
    plot_diversity_heatmap(json_file, generation='average')
    
    # Correlation heatmaps for different metrics
    print("\nGenerating correlation heatmaps...")
    
    # qbadic correlation matrix (all bases and multipliers)
    plot_correlation_heatmap(json_file, metric='qbadic', generation='average')
    
    # Other metrics (will create 1x1 heatmaps showing the single correlation value)
    # Uncomment the metrics you have tracked:
    # plot_correlation_heatmap(json_file, metric='euclidean', generation='average')
    # plot_correlation_heatmap(json_file, metric='padic', generation='average')