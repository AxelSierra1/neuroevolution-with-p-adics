import numpy as np
import json
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, kendalltau

class EvolutionMetrics:
    def __init__(self, save_dir='metrics', metrics=None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Default metrics if none provided
        if metrics is None:
            metrics = ['euclidean', 'padic']
        self.metrics = metrics
        
        # Initialize history with dynamic metric keys
        self.history = {
            'generation': [],
            'best_fitness': [],
            'mean_fitness': [],
            'worst_fitness': [],
        }
        
        # Add metric-specific keys dynamically
        for metric in self.metrics:
            metric_key = metric.replace('-', '')
            self.history[f'{metric_key}_diversity'] = []
            self.history[f'{metric_key}_to_best'] = []
            self.history[f'fitness_vs_{metric_key}'] = []
        
        # Store p-adic configuration
        self.padic_config = {}
    
    # Record metrics for a generation
    def record_generation(self, generation, population, padic_norm, p=11, precision=3):
        from nn_reals.Population import Population
        
        # Store p-adic configuration (only on first call, if p-adic is used)
        if 'padic' in self.metrics and not self.padic_config:
            self.padic_config = {'p': p, 'precision': precision, 'norm': padic_norm}
        
        fitnesses = population.get_fitnesses()
        best_net = population.get_best_networks(n=1)
        
        # Basic fitness stats
        self.history['generation'].append(generation)
        self.history['best_fitness'].append(np.min(fitnesses))
        self.history['mean_fitness'].append(np.mean(fitnesses))
        self.history['worst_fitness'].append(np.max(fitnesses))
        
        # Diversity for each metric
        for metric in self.metrics:
            if metric == 'padic':
                diversity = self._compute_padic_diversity(population, p, precision, padic_norm)
            else:
                diversity = population.population_diversity(n_samples=100, metric=metric)
            
            metric_key = metric.replace('-', '')
            self.history[f'{metric_key}_diversity'].append(diversity['mean_distance'])
        
        # Distance to best individual for all metrics
        distances_to_best = {metric: [] for metric in self.metrics}
        fitness_diffs = []
        
        # Sample subset to avoid too much computation
        sample_size = min(50, population.pop_size)
        sample_indices = np.random.choice(population.pop_size, sample_size, replace=False)
        
        for idx in sample_indices:
            net = population[idx]
            if net is not best_net:
                fitness_diffs.append(abs(net.fitness() - best_net.fitness()))
                
                for metric in self.metrics:
                    if metric == 'padic':
                        dist = Population.genetic_distance(best_net, net, metric='padic', 
                                                          p=p, precision=precision, padic_norm=padic_norm)
                    else:
                        dist = Population.genetic_distance(best_net, net, metric=metric)
                    distances_to_best[metric].append(dist)
        
        # Store mean distances to best
        for metric in self.metrics:
            metric_key = metric.replace('-', '')
            self.history[f'{metric_key}_to_best'].append(np.mean(distances_to_best[metric]))
        
        # Store correlation data for each metric
        # Format: (fitness_diff, metric1_dist, metric2_dist, ...)
        for i, metric in enumerate(self.metrics):
            metric_key = metric.replace('-', '')
            correlation_data = list(zip(fitness_diffs, distances_to_best[metric]))
            self.history[f'fitness_vs_{metric_key}'].append(correlation_data)
    
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
            
            dist = Population.genetic_distance(net1, net2, metric='padic', 
                                              p=p, precision=precision, padic_norm=padic_norm)
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
        
        # Create a copy of history with configuration included
        save_data = {
            'metrics_tracked': self.metrics,
            'padic_config': self.padic_config if self.padic_config else None,
            'metrics': self.history
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"Metrics saved to {filepath}")
    
    # methods: pearson, spearman, kendall
    def get_correlations(self, generation, metric=None, method='pearson'):
        """
        Get correlations for a specific generation.
        If metric is None, returns correlations for all metrics.
        """
        if generation >= len(self.history['generation']):
            return None
        
        results = {}
        metrics_to_check = [metric] if metric else self.metrics
        
        for m in metrics_to_check:
            metric_key = m.replace('-', '')
            data = self.history[f'fitness_vs_{metric_key}'][generation]
            
            if not data:
                continue
                
            fitness_diffs = [d[0] for d in data]
            metric_dists = [d[1] for d in data]
            
            if method == 'pearson':
                corr = pearsonr(fitness_diffs, metric_dists)[0]
            elif method == 'spearman':
                corr = spearmanr(fitness_diffs, metric_dists)[0]
            elif method == 'kendall':
                corr = kendalltau(fitness_diffs, metric_dists)[0]
            else:
                raise ValueError(f"Unknown method: {method}")
            
            results[f'{metric_key}_correlation'] = corr
        
        return results
    
    def summary_report(self):
        """Print summary statistics across all generations"""
        print("\n" + "="*60)
        print("EVOLUTION METRICS SUMMARY")
        print("="*60)
        
        print(f"\nMetrics tracked: {', '.join(self.metrics)}")
        
        if self.padic_config:
            print(f"\nP-adic configuration:")
            print(f"  Prime (p): {self.padic_config.get('p', 'N/A')}")
            print(f"  Precision: {self.padic_config.get('precision', 'N/A')}")
            print(f"  Norm type: {self.padic_config.get('norm', 'N/A')}")
        
        print(f"\nTotal generations: {len(self.history['generation'])}")
        print(f"Best fitness achieved: {min(self.history['best_fitness']):.6f}")
        print(f"Final fitness: {self.history['best_fitness'][-1]:.6f}")
        
        print("\nDiversity trends:")
        for metric in self.metrics:
            metric_key = metric.replace('-', '')
            diversity_key = f'{metric_key}_diversity'
            if diversity_key in self.history and self.history[diversity_key]:
                print(f"  {metric.capitalize():12s} - Start: {self.history[diversity_key][0]:.4f}, "
                      f"End: {self.history[diversity_key][-1]:.4f}")
        
        # Average correlations across middle generations (avoid early noise)
        mid_start = len(self.history['generation']) // 4
        mid_end = 3 * len(self.history['generation']) // 4
        
        metric_corrs = {metric: [] for metric in self.metrics}
        
        for gen in range(mid_start, mid_end):
            corr = self.get_correlations(gen)
            if corr:
                for metric in self.metrics:
                    metric_key = metric.replace('-', '')
                    corr_key = f'{metric_key}_correlation'
                    if corr_key in corr and not np.isnan(corr[corr_key]):
                        metric_corrs[metric].append(corr[corr_key])
        
        if any(metric_corrs.values()):
            print(f"\nAverage correlations (fitness diff vs distance):")
            for metric in self.metrics:
                if metric_corrs[metric]:
                    print(f"  {metric.capitalize():12s}: {np.mean(metric_corrs[metric]):.4f}")
        
        print("="*60 + "\n")