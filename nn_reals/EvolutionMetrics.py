import numpy as np
import json
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, kendalltau

class EvolutionMetrics:
    def __init__(self, save_dir='metrics', metrics=None, precisions=None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Default metrics if none provided
        if metrics is None:
            metrics = ['euclidean', 'padic']
        self.metrics = metrics
        
        # Default precisions for p-adic metrics
        if precisions is None:
            precisions = [2, 3, 4, 5]
        self.precisions = precisions
        
        # Initialize history with dynamic metric keys
        self.history = {
            'generation': [],
            'best_fitness': [],
            'mean_fitness': [],
            'worst_fitness': [],
        }
        
        # Add metric-specific keys for non-padic metrics
        for metric in self.metrics:
            if metric != 'padic':
                metric_key = metric.replace('-', '')
                self.history[f'{metric_key}_diversity'] = []
                self.history[f'{metric_key}_to_best'] = []
                self.history[f'fitness_vs_{metric_key}'] = []
        
        # Add precision-specific keys for padic metric
        if 'padic' in self.metrics:
            for prec in self.precisions:
                self.history[f'padic_prec{prec}_diversity'] = []
                self.history[f'padic_prec{prec}_to_best'] = []
                self.history[f'fitness_vs_padic_prec{prec}'] = []
        
        # Store p-adic configuration
        self.padic_config = {}
    
    def record_generation(self, generation, population, padic_norm, p=2):
        """Record metrics for a generation across all precisions"""
        from nn_reals.Population import Population
        
        # Store p-adic configuration (only on first call)
        if 'padic' in self.metrics and not self.padic_config:
            self.padic_config = {
                'p': p, 
                'precisions': self.precisions, 
                'norm': padic_norm
            }
        
        fitnesses = population.get_fitnesses()
        best_net = population.get_best_networks(n=1)
        
        # Basic fitness stats
        self.history['generation'].append(generation)
        self.history['best_fitness'].append(np.min(fitnesses))
        self.history['mean_fitness'].append(np.mean(fitnesses))
        self.history['worst_fitness'].append(np.max(fitnesses))
        
        # Process non-padic metrics
        for metric in self.metrics:
            if metric != 'padic':
                diversity = population.population_diversity(n_samples=100, metric=metric)
                metric_key = metric.replace('-', '')
                self.history[f'{metric_key}_diversity'].append(diversity['mean_distance'])
        
        # Process padic metrics for each precision
        if 'padic' in self.metrics:
            for precision in self.precisions:
                diversity = self._compute_padic_diversity(
                    population, p, precision, padic_norm
                )
                self.history[f'padic_prec{precision}_diversity'].append(
                    diversity['mean_distance']
                )
        
        # Distance to best individual for all metrics
        sample_size = min(50, population.pop_size)
        sample_indices = np.random.choice(population.pop_size, sample_size, replace=False)
        
        # Initialize distance collections
        distances_to_best = {}
        for metric in self.metrics:
            if metric == 'padic':
                for prec in self.precisions:
                    distances_to_best[f'padic_prec{prec}'] = []
            else:
                distances_to_best[metric] = []
        
        fitness_diffs = []
        
        for idx in sample_indices:
            net = population[idx]
            if net is not best_net:
                fitness_diffs.append(abs(net.fitness() - best_net.fitness()))
                
                # Non-padic metrics
                for metric in self.metrics:
                    if metric != 'padic':
                        dist = Population.genetic_distance(best_net, net, metric=metric)
                        distances_to_best[metric].append(dist)
                
                # Padic metrics for each precision
                if 'padic' in self.metrics:
                    for precision in self.precisions:
                        dist = Population.genetic_distance(
                            best_net, net, metric='padic',
                            p=p, precision=precision, padic_norm=padic_norm
                        )
                        distances_to_best[f'padic_prec{precision}'].append(dist)
        
        # Store mean distances to best
        for metric in self.metrics:
            if metric == 'padic':
                for prec in self.precisions:
                    key = f'padic_prec{prec}'
                    self.history[f'{key}_to_best'].append(
                        np.mean(distances_to_best[key])
                    )
            else:
                metric_key = metric.replace('-', '')
                self.history[f'{metric_key}_to_best'].append(
                    np.mean(distances_to_best[metric])
                )
        
        # Store correlation data
        for metric in self.metrics:
            if metric == 'padic':
                for prec in self.precisions:
                    key = f'padic_prec{prec}'
                    correlation_data = list(zip(
                        fitness_diffs, distances_to_best[key]
                    ))
                    self.history[f'fitness_vs_{key}'].append(correlation_data)
            else:
                metric_key = metric.replace('-', '')
                correlation_data = list(zip(
                    fitness_diffs, distances_to_best[metric]
                ))
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
            
            dist = Population.genetic_distance(
                net1, net2, metric='padic',
                p=p, precision=precision, padic_norm=padic_norm
            )
            distances.append(dist)
        
        distances = np.array(distances)
        
        return {
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances)
        }
    
    def save(self, filename='multi_precision_metrics.json'):
        """Save metrics to JSON file"""
        filepath = self.save_dir / filename
        
        save_data = {
            'metrics_tracked': self.metrics,
            'precisions': self.precisions if 'padic' in self.metrics else None,
            'padic_config': self.padic_config if self.padic_config else None,
            'metrics': self.history
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"Metrics saved to {filepath}")
    
    def get_correlations(self, generation, metric=None, precision=None, method='pearson'):
        """
        Get correlations for a specific generation.
        For padic metrics, specify precision. If None, returns all precisions.
        Returns None for correlations that can't be computed (constant arrays, etc.)
        """
        if generation >= len(self.history['generation']):
            return None
        
        results = {}
        
        def compute_correlation(fitness_diffs, metric_dists, method):
            """Helper to safely compute correlation with proper checks"""
            # Check if we have enough data points
            if len(fitness_diffs) < 3:
                return None
            
            # Check for constant arrays
            if len(set(fitness_diffs)) == 1 or len(set(metric_dists)) == 1:
                return None
            
            # Check for valid values (no NaN/inf)
            if any(np.isnan(fitness_diffs)) or any(np.isnan(metric_dists)):
                return None
            if any(np.isinf(fitness_diffs)) or any(np.isinf(metric_dists)):
                return None
            
            try:
                if method == 'pearson':
                    corr, _ = pearsonr(fitness_diffs, metric_dists)
                elif method == 'spearman':
                    corr, _ = spearmanr(fitness_diffs, metric_dists)
                elif method == 'kendall':
                    corr, _ = kendalltau(fitness_diffs, metric_dists)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                # Check if result is valid
                if np.isnan(corr) or np.isinf(corr):
                    return None
                    
                return corr
            except:
                return None
        
        if metric == 'padic' or (metric is None and 'padic' in self.metrics):
            precs_to_check = [precision] if precision else self.precisions
            for prec in precs_to_check:
                key = f'padic_prec{prec}'
                data = self.history[f'fitness_vs_{key}'][generation]
                
                if data and len(data) > 0:
                    fitness_diffs = [d[0] for d in data]
                    metric_dists = [d[1] for d in data]
                    
                    corr = compute_correlation(fitness_diffs, metric_dists, method)
                    if corr is not None:
                        results[f'{key}_correlation'] = corr
        
        # Handle non-padic metrics
        if metric and metric != 'padic':
            metric_key = metric.replace('-', '')
            data = self.history[f'fitness_vs_{metric_key}'][generation]
            
            if data and len(data) > 0:
                fitness_diffs = [d[0] for d in data]
                metric_dists = [d[1] for d in data]
                
                corr = compute_correlation(fitness_diffs, metric_dists, method)
                if corr is not None:
                    results[f'{metric_key}_correlation'] = corr
        elif metric is None:
            # Process all non-padic metrics
            for m in self.metrics:
                if m != 'padic':
                    metric_key = m.replace('-', '')
                    data = self.history[f'fitness_vs_{metric_key}'][generation]
                    
                    if data and len(data) > 0:
                        fitness_diffs = [d[0] for d in data]
                        metric_dists = [d[1] for d in data]
                        
                        corr = compute_correlation(fitness_diffs, metric_dists, method)
                        if corr is not None:
                            results[f'{metric_key}_correlation'] = corr
        
        return results
    
    def summary_report(self):
        """Print summary statistics across all generations"""
        print("\n" + "="*70)
        print("MULTI-PRECISION EVOLUTION METRICS SUMMARY")
        print("="*70)
        
        print(f"\nMetrics tracked: {', '.join(self.metrics)}")
        
        if self.padic_config:
            print(f"\nP-adic configuration:")
            print(f"  Prime (p): {self.padic_config.get('p', 'N/A')}")
            print(f"  Precisions: {self.padic_config.get('precisions', 'N/A')}")
            print(f"  Norm type: {self.padic_config.get('norm', 'N/A')}")
        
        print(f"\nTotal generations: {len(self.history['generation'])}")
        print(f"Best fitness achieved: {min(self.history['best_fitness']):.6f}")
        print(f"Final fitness: {self.history['best_fitness'][-1]:.6f}")
        
        print("\nDiversity trends:")
        
        # Non-padic metrics
        for metric in self.metrics:
            if metric != 'padic':
                metric_key = metric.replace('-', '')
                diversity_key = f'{metric_key}_diversity'
                if diversity_key in self.history and self.history[diversity_key]:
                    print(f"  {metric.capitalize():20s} - Start: {self.history[diversity_key][0]:.4f}, "
                          f"End: {self.history[diversity_key][-1]:.4f}")
        
        # Padic metrics by precision
        if 'padic' in self.metrics:
            for prec in self.precisions:
                key = f'padic_prec{prec}_diversity'
                if key in self.history and self.history[key]:
                    print(f"  P-adic (prec={prec}):{'':6s} - Start: {self.history[key][0]:.4f}, "
                          f"End: {self.history[key][-1]:.4f}")
        
        # Average correlations
        mid_start = len(self.history['generation']) // 4
        mid_end = 3 * len(self.history['generation']) // 4
        
        metric_corrs = {}
        for metric in self.metrics:
            if metric == 'padic':
                for prec in self.precisions:
                    metric_corrs[f'padic_prec{prec}'] = []
            else:
                metric_corrs[metric] = []
        
        valid_generations = 0
        for gen in range(mid_start, mid_end):
            corr = self.get_correlations(gen)
            if corr:
                valid_generations += 1
                for key, value in corr.items():
                    metric_name = key.replace('_correlation', '')
                    if metric_name in metric_corrs:
                        metric_corrs[metric_name].append(value)
        
        if any(metric_corrs.values()):
            print(f"\nAverage correlations (fitness diff vs distance):")
            print(f"  Based on {valid_generations}/{mid_end - mid_start} valid generations")
            
            for metric in self.metrics:
                if metric == 'padic':
                    for prec in self.precisions:
                        key = f'padic_prec{prec}'
                        if metric_corrs[key]:
                            print(f"  P-adic (prec={prec}):{'':6s} {np.mean(metric_corrs[key]):.4f} "
                                  f"({len(metric_corrs[key])} samples)")
                else:
                    if metric_corrs[metric]:
                        print(f"  {metric.capitalize():20s} {np.mean(metric_corrs[metric]):.4f} "
                              f"({len(metric_corrs[metric])} samples)")
        else:
            print("\nNo valid correlations could be computed (possibly due to constant arrays)")
        
        print("="*70 + "\n")
    
    def compare_precisions_report(self):
        """Generate a focused report comparing different precisions"""
        if 'padic' not in self.metrics:
            print("No p-adic metrics to compare")
            return
        
        print("\n" + "="*70)
        print("P-ADIC PRECISION COMPARISON")
        print("="*70)
        
        # Diversity comparison
        print("\nFinal diversity by precision:")
        for prec in self.precisions:
            key = f'padic_prec{prec}_diversity'
            if self.history[key]:
                print(f"  Precision {prec}: {self.history[key][-1]:.6f}")
        
        # Correlation comparison
        print("\nAverage correlation (mid-evolution) by precision:")
        mid_start = len(self.history['generation']) // 4
        mid_end = 3 * len(self.history['generation']) // 4
        
        for prec in self.precisions:
            corrs = []
            valid_count = 0
            for gen in range(mid_start, mid_end):
                corr = self.get_correlations(gen, precision=prec)
                if corr:
                    key = f'padic_prec{prec}_correlation'
                    if key in corr:
                        valid_count += 1
                        corrs.append(corr[key])
            
            if corrs:
                print(f"  Precision {prec}: {np.mean(corrs):.4f} "
                      f"(σ={np.std(corrs):.4f}, n={valid_count})")
            else:
                print(f"  Precision {prec}: No valid correlations")
        
        print("="*70 + "\n")