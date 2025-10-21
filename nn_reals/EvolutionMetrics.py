import numpy as np
import json
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, kendalltau

class EvolutionMetrics:
    def __init__(self, save_dir='metrics', metrics=None, qpadic_primes=None, multipliers=None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.metrics = metrics or ['euclidean', 'qpadic']
        self.qpadic_primes = qpadic_primes or [2, 3, 5, 7]
        self.multipliers = multipliers or [2, 3, 4, 5]
        self.qpadic_config = {}
        
        # Initialize history
        self.history = {
            'generation': [], 'best_fitness': [], 
            'mean_fitness': [], 'worst_fitness': []
        }
        
        # Add metric-specific keys
        for metric in self.metrics:
            if metric == 'qpadic':
                for p in self.qpadic_primes:
                    for m in self.multipliers:
                        prefix = f'qpadic_p{p}_mult{m}'
                        for suffix in ['diversity', 'to_best', '']:
                            key = f'{prefix}_{suffix}' if suffix else f'fitness_vs_{prefix}'
                            self.history[key] = []
            else:
                key = metric.replace('-', '')
                for suffix in ['diversity', 'to_best', '']:
                    self.history[f'{key}_{suffix}' if suffix else f'fitness_vs_{key}'] = []
    
    def record_generation(self, generation, population, qpadic_norm):
        """Record metrics for a generation"""
        from nn_reals.Population import Population
        
        # Store config on first call
        if 'qpadic' in self.metrics and not self.qpadic_config:
            self.qpadic_config = {
                'primes': self.qpadic_primes,
                'multipliers': self.multipliers,
                'norm': qpadic_norm
            }
        
        fitnesses = population.get_fitnesses()
        best_net = population.get_best_networks(n=1)
        
        # Basic stats
        self.history['generation'].append(generation)
        self.history['best_fitness'].append(np.min(fitnesses))
        self.history['mean_fitness'].append(np.mean(fitnesses))
        self.history['worst_fitness'].append(np.max(fitnesses))
        
        # Diversity for all metrics
        for metric in self.metrics:
            if metric == 'qpadic':
                for p in self.qpadic_primes:
                    for m in self.multipliers:
                        div = self._compute_qpadic_diversity(population, p, m, qpadic_norm)
                        self.history[f'qpadic_p{p}_mult{m}_diversity'].append(div['mean_distance'])
            else:
                div = population.population_diversity(n_samples=100, metric=metric)
                self.history[f"{metric.replace('-', '')}_diversity"].append(div['mean_distance'])
        
        # Distance to best
        sample_size = min(50, population.pop_size)
        sample_indices = np.random.choice(population.pop_size, sample_size, replace=False)
        
        distances = {m: [] for m in self.metrics if m != 'qpadic'}
        if 'qpadic' in self.metrics:
            distances.update({f'qpadic_p{p}_mult{m}': [] 
                            for p in self.qpadic_primes for m in self.multipliers})
        
        fitness_diffs = []
        
        for idx in sample_indices:
            net = population[idx]
            if net is not best_net:
                fitness_diffs.append(abs(net.fitness() - best_net.fitness()))
                
                for metric in self.metrics:
                    if metric == 'qpadic':
                        for p in self.qpadic_primes:
                            for m in self.multipliers:
                                dist = Population.genetic_distance(
                                    best_net, net, metric='qpadic',
                                    p=p, multiplier=m, qpadic_norm=qpadic_norm
                                )
                                distances[f'qpadic_p{p}_mult{m}'].append(dist)
                    else:
                        dist = Population.genetic_distance(best_net, net, metric=metric)
                        distances[metric].append(dist)
        
        # Store means and correlations
        for key, dists in distances.items():
            metric_key = key if 'qpadic' in key else key.replace('-', '')
            self.history[f'{metric_key}_to_best'].append(np.mean(dists))
            self.history[f'fitness_vs_{metric_key}'].append(list(zip(fitness_diffs, dists)))
    
    def _compute_qpadic_diversity(self, population, p, multiplier, qpadic_norm, n_samples=100):
        """Compute qp-adic diversity"""
        if population.pop_size < 2:
            raise ValueError("Population must have at least 2 networks")
        
        from nn_reals.Population import Population
        
        n_samples = min(n_samples, population.pop_size * (population.pop_size - 1) // 2)
        distances = []
        
        for _ in range(n_samples):
            idx1, idx2 = np.random.choice(population.pop_size, 2, replace=False)
            dist = Population.genetic_distance(
                population[idx1], population[idx2], metric='qpadic',
                p=p, multiplier=multiplier, qpadic_norm=qpadic_norm
            )
            distances.append(dist)
        
        distances = np.array(distances)
        return {
            'mean_distance': np.mean(distances), 'std_distance': np.std(distances),
            'min_distance': np.min(distances), 'max_distance': np.max(distances)
        }
    
    def save(self, filename='multi_prime_metrics.json'):
        """Save metrics to JSON"""
        save_data = {
            'metrics_tracked': self.metrics,
            'qpadic_primes': self.qpadic_primes if 'qpadic' in self.metrics else None,
            'multipliers': self.multipliers if 'qpadic' in self.metrics else None,
            'qpadic_config': self.qpadic_config or None,
            'metrics': self.history
        }
        
        with open(self.save_dir / filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"Metrics saved to {self.save_dir / filename}")
    
    def get_correlations(self, generation, metric=None, prime=None, multiplier=None, method='pearson'):
        """Get correlations for a generation"""
        if generation >= len(self.history['generation']):
            return None
        
        def safe_corr(x, y):
            """Safely compute correlation"""
            if len(x) < 3 or len(set(x)) == 1 or len(set(y)) == 1:
                return None
            if any(np.isnan(x)) or any(np.isinf(x)) or any(np.isnan(y)) or any(np.isinf(y)):
                return None
            
            try:
                corr_func = {'pearson': pearsonr, 'spearman': spearmanr, 'kendall': kendalltau}[method]
                corr, _ = corr_func(x, y)
                return None if np.isnan(corr) or np.isinf(corr) else corr
            except:
                return None
        
        results = {}
        
        # Handle qpadic
        if metric == 'qpadic' or (metric is None and 'qpadic' in self.metrics):
            for p in ([prime] if prime else self.qpadic_primes):
                for m in ([multiplier] if multiplier else self.multipliers):
                    key = f'qpadic_p{p}_mult{m}'
                    data = self.history[f'fitness_vs_{key}'][generation]
                    if data:
                        x, y = zip(*data)
                        corr = safe_corr(x, y)
                        if corr is not None:
                            results[f'{key}_correlation'] = corr
        
        # Handle non-qpadic
        metrics_to_check = [metric] if metric and metric != 'qpadic' else [m for m in self.metrics if m != 'qpadic']
        for m in metrics_to_check:
            key = m.replace('-', '')
            data = self.history[f'fitness_vs_{key}'][generation]
            if data:
                x, y = zip(*data)
                corr = safe_corr(x, y)
                if corr is not None:
                    results[f'{key}_correlation'] = corr
        
        return results
    
    def summary_report(self):
        """Print summary statistics"""
        print("\n" + "="*70)
        print("MULTI-PRIME EVOLUTION METRICS SUMMARY")
        print("="*70)
        print(f"\nMetrics tracked: {', '.join(self.metrics)}")
        
        if self.qpadic_config:
            print(f"\nqp-adic configuration:")
            print(f"  Primes: {self.qpadic_config['primes']}")
            print(f"  Precisions: {self.qpadic_config['multipliers']}")
            print(f"  Norm type: {self.qpadic_config['norm']}")
        
        print(f"\nTotal generations: {len(self.history['generation'])}")
        print(f"Best fitness: {min(self.history['best_fitness']):.6f}")
        print(f"Final fitness: {self.history['best_fitness'][-1]:.6f}")
        
        print("\nDiversity trends:")
        for metric in self.metrics:
            if metric == 'qpadic':
                for p in self.qpadic_primes:
                    for m in self.multipliers:
                        key = f'qpadic_p{p}_mult{m}_diversity'
                        if self.history[key]:
                            print(f"  qp-adic (p={p}, m={m}): Start: {self.history[key][0]:.4f}, End: {self.history[key][-1]:.4f}")
            else:
                key = f"{metric.replace('-', '')}_diversity"
                if self.history[key]:
                    print(f"  {metric.capitalize():20s} - Start: {self.history[key][0]:.4f}, End: {self.history[key][-1]:.4f}")
        
        # Correlations
        mid_start, mid_end = len(self.history['generation']) // 4, 3 * len(self.history['generation']) // 4
        corrs = {f'qpadic_p{p}_mult{m}': [] for p in self.qpadic_primes for m in self.multipliers} if 'qpadic' in self.metrics else {}
        corrs.update({m: [] for m in self.metrics if m != 'qpadic'})
        
        valid_gens = 0
        for gen in range(mid_start, mid_end):
            c = self.get_correlations(gen)
            if c:
                valid_gens += 1
                for k, v in c.items():
                    metric_name = k.replace('_correlation', '')
                    if metric_name in corrs:
                        corrs[metric_name].append(v)
        
        if any(corrs.values()):
            print(f"\nAverage correlations (fitness diff vs distance):")
            print(f"  Based on {valid_gens}/{mid_end - mid_start} valid generations")
            for metric in self.metrics:
                if metric == 'qpadic':
                    for p in self.qpadic_primes:
                        for m in self.multipliers:
                            key = f'qpadic_p{p}_mult{m}'
                            if corrs[key]:
                                print(f"  qp-adic (p={p}, m={m}): {np.mean(corrs[key]):.4f} ({len(corrs[key])} samples)")
                elif corrs[metric]:
                    print(f"  {metric.capitalize():20s} {np.mean(corrs[metric]):.4f} ({len(corrs[metric])} samples)")
        else:
            print("\nNo valid correlations computed")
        print("="*70 + "\n")
    
    def compare_primes_report(self):
        """Compare different primes and multipliers"""
        if 'qpadic' not in self.metrics:
            print("No qp-adic metrics to compare")
            return
        
        print("\n" + "="*70)
        print("QP-ADIC PRIME AND PRECISION COMPARISON")
        print("="*70)
        
        print("\nFinal diversity by (prime, multiplier):")
        for p in self.qpadic_primes:
            print(f"\n  Prime {p}:")
            for m in self.multipliers:
                key = f'qpadic_p{p}_mult{m}_diversity'
                if self.history[key]:
                    print(f"    Precision {m}: {self.history[key][-1]:.6f}")
        
        print("\nAverage correlation (mid-evolution) by (prime, multiplier):")
        mid_start, mid_end = len(self.history['generation']) // 4, 3 * len(self.history['generation']) // 4
        
        for p in self.qpadic_primes:
            print(f"\n  Prime {p}:")
            for m in self.multipliers:
                corrs, valid = [], 0
                for gen in range(mid_start, mid_end):
                    c = self.get_correlations(gen, prime=p, multiplier=m)
                    key = f'qpadic_p{p}_mult{m}_correlation'
                    if c and key in c:
                        valid += 1
                        corrs.append(c[key])
                
                if corrs:
                    print(f"    Precision {m}: {np.mean(corrs):.4f} (σ={np.std(corrs):.4f}, n={valid})")
                else:
                    print(f"    Precision {m}: No valid correlations")
        print("="*70 + "\n")