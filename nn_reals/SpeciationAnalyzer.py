import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
from matplotlib.patches import Rectangle

from nn_reals.Population import Population

# Analyzes and visualizes speciation patterns in neuroevolution populations
# pop_history: Atribute of a Neuroevolution instance (list of populations across generations)
class SpeciationAnalyzer:
    def __init__(self, pop_history, distance_metric='euclidean'):
        self.pop_history = pop_history
        self.distance_metric = distance_metric
        self.species_history = []
        self.phylogeny = None # evolutionary history of the development of a species

    # Cluster networks into species based on distance threshold, returns a list of species (each species is list of network indices)
    def cluster_into_species(self, distance_matrix, threshold):
        n = distance_matrix.shape[0]
        
        if n < 2:
            return [[i for i in range(n)]], None
        
        condensed_distances = squareform(distance_matrix, checks=False) # Stores only the distance under the diagonal of the dist matrix (More efficient)
        linkage_matrix = linkage(condensed_distances, method='average') # encodes who merged with whom and at what distance
        cluster_labels = fcluster(linkage_matrix, threshold, criterion='distance')
        
        # Group networks by cluster
        species = {}
        for net_idx, species_id in enumerate(cluster_labels):
            if species_id not in species:
                species[species_id] = []
            species[species_id].append(net_idx)
        
        # print(f"DEBUG: Distance matrix shape: {distance_matrix.shape}")
        # print(f"DEBUG: Distance range: [{distance_matrix[distance_matrix > 0].min():.6f}, {distance_matrix.max():.6f}]")
        # print(f"DEBUG: Threshold: {threshold}")
        # print(f"DEBUG: Cluster labels: {cluster_labels}")
        # print(f"DEBUG: Number of species found: {len(species)}")
        
        return list(species.values()), linkage_matrix
    
    # Analyze speciation for a specific generation (gen_idx starts at 0)
    def analyze_generation(self, gen_idx, threshold):
        population = self.pop_history[gen_idx]

        # Compute pairwise distances
        distance_matrix = np.zeros((len(population), len(population)))
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                dist = Population.distance(
                    population[i], population[j], self.distance_metric
                )
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        # Cluster into species
        species, linkage_matrix = self.cluster_into_species(distance_matrix, threshold)
        
        return species, distance_matrix, linkage_matrix

    # Red = worst fitness, Green = best fitness
    def get_fitness_colors(self, population, cmap_name='RdYlGn_r'):
        fitnesses = np.array([net.fitness() for net in population])
        
        # Normalize fitnesses to [0, 1] for coloring
        fit_min, fit_max = fitnesses.min(), fitnesses.max()
        if fit_max == fit_min:
            normalized = np.ones_like(fitnesses) * 0.5
        else:
            normalized = (fitnesses - fit_min) / (fit_max - fit_min)
        cmap = plt.get_cmap(cmap_name)
        colors = cmap(normalized)
        
        return colors, fitnesses
    
    def plot_phylogenetic_tree(self, gen_idx, threshold, figsize=(16, 10)):
        species, distance_matrix, linkage_matrix = self.analyze_generation(gen_idx, threshold)
        population = self.pop_history[gen_idx]
        colors, fitnesses = self.get_fitness_colors(population)
        
        if len(population) < 2:
            print(f"Warning: Generation {gen_idx} has fewer than 2 networks. Skipping visualization.")
            return None, None
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Speciation Analysis - Generation {gen_idx}', fontsize=16, fontweight='bold')
        
        # Plot 1: Dendrogram
        ax1 = axes[0, 0]
        dendro = dendrogram(
            linkage_matrix,
            ax=ax1,
            color_threshold=threshold,
            above_threshold_color='gray'
        )
        ax1.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold}')
        ax1.set_xlabel('Network Index')
        ax1.set_ylabel('Distance')
        ax1.set_title('Hierarchical Clustering Dendrogram')
        ax1.legend()
        
        # Plot 2: Fitness distribution with species coloring
        ax2 = axes[0, 1]
        species_id_map = np.zeros(len(population), dtype=int)
        for sp_id, species_members in enumerate(species):
            for net_idx in species_members:
                species_id_map[net_idx] = sp_id
        
        scatter = ax2.scatter(
            range(len(population)),
            fitnesses,
            c=colors,
            s=100,
            edgecolors='black',
            linewidth=1
        )
        ax2.set_xlabel('Network Index')
        ax2.set_ylabel('Fitness (MSE)')
        ax2.set_title('Fitness Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Species composition
        ax3 = axes[1, 0]
        species_sizes = [len(sp) for sp in species]
        species_labels = [f'Species {i}\n(n={size})' for i, size in enumerate(species_sizes)]
        species_colors = plt.cm.tab20(np.arange(len(species)) % 20)
        
        bars = ax3.bar(range(len(species)), species_sizes, color=species_colors, edgecolor='black', linewidth=1.5)
        ax3.set_xlabel('Species ID')
        ax3.set_ylabel('Number of Networks')
        ax3.set_title(f'Species Distribution ({len(species)} species)')
        ax3.set_xticks(range(len(species)))
        
        # Plot 4: Average fitness per species
        ax4 = axes[1, 1]
        species_avg_fitness = []
        species_std_fitness = []
        for sp_members in species:
            sp_fitnesses = [fitnesses[idx] for idx in sp_members]
            species_avg_fitness.append(np.mean(sp_fitnesses))
            species_std_fitness.append(np.std(sp_fitnesses))
        
        ax4.bar(range(len(species)), species_avg_fitness, yerr=species_std_fitness,
                color=species_colors, edgecolor='black', linewidth=1.5, capsize=5)
        ax4.set_xlabel('Species ID')
        ax4.set_ylabel('Average Fitness')
        ax4.set_title('Mean Fitness per Species (± std)')
        ax4.set_xticks(range(len(species)))
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig, (species, fitnesses, distance_matrix)
    
    def plot_heatmap_with_species(self, gen_idx, threshold, figsize=(12, 10)):
        species, distance_matrix, _ = self.analyze_generation(gen_idx, threshold)
        population = self.pop_history[gen_idx]
        colors_array, fitnesses = self.get_fitness_colors(population)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f'Distance Matrix Heatmap - Generation {gen_idx}', fontsize=14, fontweight='bold')
        
        # Reorder distance matrix by species
        species_order = []
        species_ids = []
        for sp_id, sp_members in enumerate(species):
            species_order.extend(sp_members)
            species_ids.extend([sp_id] * len(sp_members))
        
        reordered_matrix = distance_matrix[np.ix_(species_order, species_order)]
        
        ax1 = axes[0]
        im = ax1.imshow(reordered_matrix, cmap='viridis', aspect='auto')
        ax1.set_xlabel('Network Index (reordered by species)')
        ax1.set_ylabel('Network Index (reordered by species)')
        ax1.set_title('Pairwise Distance Matrix')
        
        # Draw species boundaries
        boundary = 0
        for sp_id, sp_members in enumerate(species):
            boundary += len(sp_members)
            ax1.axhline(y=boundary - 0.5, color='red', linewidth=2, linestyle='--')
            ax1.axvline(x=boundary - 0.5, color='red', linewidth=2, linestyle='--')
        
        plt.colorbar(im, ax=ax1, label='Distance')
        
        # Fitness distribution by species
        ax2 = axes[1]
        for sp_id, sp_members in enumerate(species):
            sp_fitnesses = fitnesses[sp_members]
            y_positions = np.full(len(sp_fitnesses), sp_id) + np.random.normal(0, 0.04, len(sp_fitnesses))
            sp_colors = colors_array[sp_members]
            ax2.scatter(sp_fitnesses, y_positions, c=sp_colors, s=100,
                       edgecolors='black', linewidth=1, alpha=0.7)
        
        ax2.set_xlabel('Fitness (MSE)')
        ax2.set_ylabel('Species ID')
        ax2.set_title('Fitness Distribution within Species')
        ax2.set_yticks(range(len(species)))
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig
    
    def plot_speciation_over_time(self, threshold, figsize=(14, 6)):
        """Track speciation metrics across generations."""
        n_species = []
        avg_species_size = []
        max_species_size = []
        generations = []
        
        for gen_idx in range(len(self.pop_history)):
            species, _, _ = self.analyze_generation(gen_idx, threshold)
            n_species.append(len(species))
            avg_species_size.append(np.mean([len(s) for s in species]))
            max_species_size.append(np.max([len(s) for s in species]))
            generations.append(gen_idx)
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle(f'Speciation Over Time (threshold={threshold})', fontsize=14, fontweight='bold')
        
        axes[0].plot(generations, n_species, marker='o', linewidth=2, markersize=6)
        axes[0].set_xlabel('Generation')
        axes[0].set_ylabel('Number of Species')
        axes[0].set_title('Species Count')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(generations, avg_species_size, marker='s', linewidth=2, markersize=6, label='Average')
        axes[1].plot(generations, max_species_size, marker='^', linewidth=2, markersize=6, label='Maximum')
        axes[1].set_xlabel('Generation')
        axes[1].set_ylabel('Species Size')
        axes[1].set_title('Species Size Trends')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Population diversity
        diversity = []
        for gen_idx in range(len(self.pop_history)):
            population = self.pop_history[gen_idx]
            all_distances = []
            for i in range(min(len(population), 50)):
                for j in range(i + 1, min(len(population), 50)):
                    dist = Population.distance(population[i], population[j], self.distance_metric)
                    all_distances.append(dist)
            diversity.append(np.mean(all_distances) if all_distances else 0)
        
        axes[2].plot(generations, diversity, marker='d', linewidth=2, markersize=6, color='purple')
        axes[2].set_xlabel('Generation')
        axes[2].set_ylabel('Mean Pairwise Distance')
        axes[2].set_title('Population Diversity')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_species_summary(self, gen_idx, threshold):
        species, distance_matrix, _ = self.analyze_generation(gen_idx, threshold)
        population = self.pop_history[gen_idx]
        _, fitnesses = self.get_fitness_colors(population)
        
        summary = f"\n{'='*60}\n"
        summary += f"Generation {gen_idx} - Species Summary\n"
        summary += f"{'='*60}\n"
        summary += f"Number of species: {len(species)}\n"
        summary += f"Total population size: {len(population)}\n"
        summary += f"Distance threshold: {threshold}\n\n"
        
        for sp_id, sp_members in enumerate(species):
            sp_fitnesses = fitnesses[sp_members]
            summary += f"Species {sp_id}:\n"
            summary += f"  Size: {len(sp_members)}\n"
            summary += f"  Members: {sp_members}\n"
            summary += f"  Best fitness: {np.min(sp_fitnesses):.6f}\n"
            summary += f"  Worst fitness: {np.max(sp_fitnesses):.6f}\n"
            summary += f"  Mean fitness: {np.mean(sp_fitnesses):.6f}\n"
            summary += f"  Fitness std: {np.std(sp_fitnesses):.6f}\n\n"
        
        return summary

    # Uses the 25th, 50th, and 75th percentiles as candidate thresholds
    # 25/50/75 is reasonable heuristically, but depending on the problem, these might need tuning
    def suggest_threshold(self, gen_idx):
        population = self.pop_history[gen_idx]
        
        # Compute all pairwise distances
        distance_matrix = np.zeros((len(population), len(population)))
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                dist = Population.distance(
                    population[i], population[j], self.distance_metric
                )
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        # Get upper triangle values (excluding diagonal)
        distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
        distances = distances[distances > 0]
        
        print(f"\nDistance Statistics for Generation {gen_idx}:")
        print(f"Min distance: {distances.min():.6f}")
        print(f"Max distance: {distances.max():.6f}")
        print(f"Mean distance: {distances.mean():.6f}")
        print(f"Median distance: {np.median(distances):.6f}")
        print(f"25th percentile: {np.percentile(distances, 25):.6f}")
        print(f"75th percentile: {np.percentile(distances, 75):.6f}")
        
        suggested_thresholds = {
            'Very strict (fine-grained species)': np.percentile(distances, 25),
            'Moderate (balanced)': np.percentile(distances, 50),
            'Loose (coarse-grained species)': np.percentile(distances, 75),
        }
        
        print("\nSuggested thresholds:")
        for desc, threshold in suggested_thresholds.items():
            print(f"  {desc}: {threshold:.6f}")
        
        return suggested_thresholds