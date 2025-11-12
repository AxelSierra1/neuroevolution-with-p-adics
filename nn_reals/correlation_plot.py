import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_heatmap(json_file, generation_range='mid'):
    """Create a heatmap showing qp-adic correlation by base and multiplier."""
    # Load data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    bases = data['qpadic_bases']
    multipliers = data['multipliers']
    metrics = data['metrics']
    total_gens = len(metrics['generation'])
    
    # Determine generation range
    if generation_range == 'mid':
        start_gen = total_gens // 4
        end_gen = 3 * total_gens // 4
        range_label = f"Generaciones {metrics['generation'][start_gen]}-{metrics['generation'][end_gen-1]} (Mid-Evolution)"
    elif generation_range == 'early':
        start_gen = 0
        end_gen = total_gens // 4
        range_label = f"Generaciones {metrics['generation'][start_gen]}-{metrics['generation'][end_gen-1]} (Early)"
    elif generation_range == 'late':
        start_gen = 3 * total_gens // 4
        end_gen = total_gens
        range_label = f"Generaciones {metrics['generation'][start_gen]}-{metrics['generation'][end_gen-1]} (Late)"
    elif generation_range == 'all':
        start_gen = 0
        end_gen = total_gens
        range_label = f"Generaciones {metrics['generation'][0]}-{metrics['generation'][-1]} (All)"
    else:
        # Custom range: generation_range should be tuple (start, end)
        start_gen, end_gen = generation_range
        range_label = f"Generaciones {start_gen}-{end_gen-1}"
    
    # Build correlation matrix
    correlation_matrix = np.full((len(bases), len(multipliers)), np.nan)
    valid_counts = np.zeros((len(bases), len(multipliers)))
    
    for i, base in enumerate(bases):
        for j, mult in enumerate(multipliers):
            key = f'fitness_vs_qpadic_p{base}_mult{mult}'
            correlations = []
            
            for gen_idx in range(start_gen, end_gen):
                data_points = metrics[key][gen_idx]
                
                # Need at least 3 points for correlation
                if data_points and len(data_points) >= 3:
                    fitness_diffs, distances = zip(*data_points)
                    
                    # Check for valid data
                    if len(set(fitness_diffs)) > 1 and len(set(distances)) > 1:
                        if not any(np.isnan(fitness_diffs)) and not any(np.isnan(distances)):
                            corr = np.corrcoef(fitness_diffs, distances)[0, 1]
                            if not np.isnan(corr) and not np.isinf(corr):
                                correlations.append(corr)
            
            if correlations:
                correlation_matrix[i, j] = np.mean(correlations)
                valid_counts[i, j] = len(correlations)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use a diverging colormap centered at 0
    sns.heatmap(correlation_matrix, 
                annot=False, 
                fmt='.3f',
                cmap='YlGnBu',
                vmin=0.15,
                vmax=0.26,
                xticklabels=multipliers,
                yticklabels=bases,
                cbar_kws={'label': 'Correlación (Fitness vs Distancia)'},
                ax=ax,
                mask=np.isnan(correlation_matrix))
    ax.invert_yaxis()
    
    ax.set_xlabel('Multiplicador', fontsize=12, fontweight='bold')
    ax.set_ylabel('Base', fontsize=12, fontweight='bold')
    ax.set_title(f'Correlación por base y multiplicador del Funcional de valoración b-ádico\n{range_label}', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    filename = f'correlation_heatmap_{generation_range}.png' if isinstance(generation_range, str) else f'correlation_heatmap_{start_gen}_{end_gen}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved as '{filename}'")
    plt.show()
    
    # Print summary statistics
    valid_mask = ~np.isnan(correlation_matrix)
    if valid_mask.any():
        print(f"\n{range_label} Correlation Statistics:")
        print(f"  Min correlation: {np.nanmin(correlation_matrix):.6f}")
        print(f"  Max correlation: {np.nanmax(correlation_matrix):.6f}")
        print(f"  Mean correlation: {np.nanmean(correlation_matrix):.6f}")
        print(f"  Median correlation: {np.nanmedian(correlation_matrix):.6f}")
        
        best_idx = np.unravel_index(np.nanargmax(correlation_matrix), correlation_matrix.shape)
        worst_idx = np.unravel_index(np.nanargmin(correlation_matrix), correlation_matrix.shape)
        
        print(f"\nStrongest positive correlation: Base={bases[best_idx[0]]}, "
              f"Multiplier={multipliers[best_idx[1]]} ({correlation_matrix[best_idx]:.6f})")
        print(f"  Valid generations: {int(valid_counts[best_idx])}/{end_gen-start_gen}")
        
        print(f"Strongest negative correlation: Base={bases[worst_idx[0]]}, "
              f"Multiplier={multipliers[worst_idx[1]]} ({correlation_matrix[worst_idx]:.6f})")
        print(f"  Valid generations: {int(valid_counts[worst_idx])}/{end_gen-start_gen}")
        
        print(f"\nValid data points across all configurations:")
        print(f"  Total valid: {np.sum(valid_counts):.0f}")
        print(f"  Mean per config: {np.mean(valid_counts[valid_mask]):.1f}")
    else:
        print("\nNo valid correlations found in the specified generation range.")

if __name__ == "__main__":
    # Plot correlation heatmap for mid-evolution
    plot_correlation_heatmap('metrics/run_2000gen.json', generation_range='all')
    
    # Optional: plot for different ranges
    # plot_correlation_heatmap('metrics/run_2000gen.json', generation_range='early')
    # plot_correlation_heatmap('metrics/run_2000gen.json', generation_range='late')
    # plot_correlation_heatmap('metrics/run_2000gen.json', generation_range='all')