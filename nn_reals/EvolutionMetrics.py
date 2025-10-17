import numpy as np
import json
from pathlib import Path
from scipy.stats import spearmanr, kendalltau, pearsonr

class EvolutionMetrics:
    def __init__(self, save_dir='metrics'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.history = {
            'generation': [],
            'best_fitness': [],
            'mean_fitness': [],
            'worst_fitness': [],
            'euclidean_diversity': [],
            'padic_diversity': [],
            'euclidean_to_best': [],
            'padic_to_best': [],
            'fitness_vs_euclidean': [],
            'fitness_vs_padic': []
        }
        
        # Store p-adic configuration
        self.padic_config = {}
    
    # Record metrics for a generation
    def record_generation(self, generation, population, padic_norm, metric_pairs=['euclidean', 'padic'], p=11, precision=3):
        from nn_reals.Population import Population
        
        # Store p-adic configuration (only on first call)
        if not self.padic_config:
            self.padic_config = {'p': p, 'precision': precision, 'norm': padic_norm}
        
        fitnesses = population.get_fitnesses()
        best_net = population.get_best_networks(n=1)
        
        # Basic fitness stats
        self.history['generation'].append(generation)
        self.history['best_fitness'].append(np.min(fitnesses))
        self.history['mean_fitness'].append(np.mean(fitnesses))
        self.history['worst_fitness'].append(np.max(fitnesses))
        
        # Diversity for each metric
        diversity_results = {}
        for metric in metric_pairs:
            if metric == 'padic':
                diversity = self._compute_padic_diversity(population, p, precision, padic_norm)
            else:
                diversity = population.population_diversity(n_samples=100, metric=metric)
            
            key = f"{metric.replace('-', '')}_diversity"
            self.history[key].append(diversity['mean_distance'])
            diversity_results[metric] = diversity
        
        # Distance to best individual
        distances_to_best = {'euclidean': [], 'padic': []}
        fitness_diffs = []
        
        # Sample subset to avoid too much computation
        sample_size = min(50, population.pop_size)
        sample_indices = np.random.choice(population.pop_size, sample_size, replace=False)
        
        for idx in sample_indices:
            net = population[idx]
            if net is not best_net:
                fitness_diffs.append(abs(net.fitness() - best_net.fitness()))
                
                for metric in metric_pairs:
                    if metric == 'padic':
                        dist = Population.genetic_distance(best_net, net, metric='padic', p=p, precision=precision, padic_norm=padic_norm)
                    else:
                        dist = Population.genetic_distance(best_net, net, metric=metric)
                    distances_to_best[metric].append(dist)
        
        # Store mean distances to best
        for metric in metric_pairs:
            key = f"{metric.replace('-', '')}_to_best"
            self.history[key].append(np.mean(distances_to_best[metric]))
        
        # Store correlation data (fitness_diff, euclidean_dist, padic_dist)
        correlation_data = list(zip(
            fitness_diffs,
            distances_to_best.get('euclidean', [0] * len(fitness_diffs)),
            distances_to_best.get('padic', [0] * len(fitness_diffs))
        ))
        self.history['fitness_vs_euclidean'].append(correlation_data)
        self.history['fitness_vs_padic'].append(correlation_data)
    
    def _compute_padic_diversity(self, population, p, precision, padic_norm, n_samples=100):
        """Helper method to compute p-adic diversity with custom parameters."""
        if population.pop_size < 2:
            raise ValueError("Population must have at least 2 networks")
        
        from nn_reals.Population import Population
        
        max_possible_pairs = population.pop_size * (population.pop_size - 1) // 2
        n_samples = min(n_samples, max_possible_pairs)
        
        distances = []
        for _ in range(n_samples):
            idx1, idx2 = np.random.choice(population.pop_size, 2, replace=False)
            net1, net2 = population[idx1], population[idx2]
            
            dist = Population.genetic_distance(net1, net2, metric='padic', p=p, precision=precision, padic_norm=padic_norm)
            distances.append(dist)
        
        distances = np.array(distances)
        
        return {
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances)
        }
    
    def save(self, filename='evolution_metrics.json'):
        """Save metrics to JSON file"""
        filepath = self.save_dir / filename
        
        # Create a copy of history with p-adic config included
        save_data = {
            'padic_config': self.padic_config,
            'metrics': self.history
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"Metrics saved to {filepath}")
    
    # method: 'pearson', 'spearman', or 'kendall'
    def get_correlations(self, generation, method='spearman'):
        if generation >= len(self.history['generation']):
            return None
        
        data = self.history['fitness_vs_euclidean'][generation]
        fitness_diffs = [d[0] for d in data]
        euclidean_dists = [d[1] for d in data]
        padic_dists = [d[2] for d in data]
        
        if method == 'pearson':
            euclidean_corr = pearsonr(fitness_diffs, euclidean_dists)[0]
            padic_corr = pearsonr(fitness_diffs, padic_dists)[0]
        elif method == 'spearman':
            euclidean_corr = spearmanr(fitness_diffs, euclidean_dists)[0]
            padic_corr = spearmanr(fitness_diffs, padic_dists)[0]
        elif method == 'kendall':
            euclidean_corr = kendalltau(fitness_diffs, euclidean_dists)[0]
            padic_corr = kendalltau(fitness_diffs, padic_dists)[0]
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            'euclidean_correlation': euclidean_corr,
            'padic_correlation': padic_corr
        }
    
    def summary_report(self):
        """Print summary statistics across all generations"""
        print("\n" + "="*60)
        print("EVOLUTION METRICS SUMMARY")
        print("="*60)
        
        if self.padic_config:
            print(f"\nP-adic configuration:")
            print(f"  Prime (p): {self.padic_config.get('p', 'N/A')}")
            print(f"  Precision: {self.padic_config.get('precision', 'N/A')}")
            print(f"  Norm type: {self.padic_config.get('norm', 'N/A')}")
        
        print(f"\nTotal generations: {len(self.history['generation'])}")
        print(f"Best fitness achieved: {min(self.history['best_fitness']):.6f}")
        print(f"Final fitness: {self.history['best_fitness'][-1]:.6f}")
        
        print("\nDiversity trends:")
        print(f"  Euclidean - Start: {self.history['euclidean_diversity'][0]:.4f}, "
              f"End: {self.history['euclidean_diversity'][-1]:.4f}")
        print(f"  P-adic    - Start: {self.history['padic_diversity'][0]:.4f}, "
              f"End: {self.history['padic_diversity'][-1]:.4f}")
        
        # Average correlations across middle generations (avoid early noise)
        mid_start = len(self.history['generation']) // 4
        mid_end = 3 * len(self.history['generation']) // 4
        
        euclidean_corrs = []
        padic_corrs = []
        
        for gen in range(mid_start, mid_end):
            corr = self.get_correlations(gen)
            if corr and not np.isnan(corr['euclidean_correlation']):
                euclidean_corrs.append(corr['euclidean_correlation'])
                padic_corrs.append(corr['padic_correlation'])
        
        if euclidean_corrs:
            print(f"\nAverage correlations (fitness diff vs distance):")
            print(f"  Euclidean: {np.mean(euclidean_corrs):.4f}")
            print(f"  P-adic:    {np.mean(padic_corrs):.4f}")
        
        print("="*60 + "\n")