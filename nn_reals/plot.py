import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_diversity_heatmap(json_file, generation='final'):
    """Create a heatmap showing qp-adic diversity by base and multiplier."""
    # Load data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    bases = data['qpadic_bases']
    multipliers = data['multipliers']
    metrics = data['metrics']
    
    # Determine which generation to use
    if generation == 'final':
        gen_idx = -1
        gen_label = f"Generacion {metrics['generation'][-1]}"
    elif generation == 'initial':
        gen_idx = 0
        gen_label = f"Generacion {metrics['generation'][0]}"
    else:
        gen_idx = metrics['generation'].index(generation)
        gen_label = f"Generacion {generation}"
    
    # Build diversity matrix
    diversity_matrix = np.zeros((len(bases), len(multipliers)))
    
    for i, base in enumerate(bases):
        for j, mult in enumerate(multipliers):
            key = f'qpadic_p{base}_mult{mult}_diversity'
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
    plt.savefig(f'diversity_heatmap_{generation}.png', dpi=300, bbox_inches='tight')
    print(f"Heatmap saved as 'diversity_heatmap_{generation}.png'")
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
    # Single heatmap for final generation
    plot_diversity_heatmap('metrics/run_1000gen.json', generation=10)