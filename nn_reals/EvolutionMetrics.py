import numpy as np
import json
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, kendalltau

class EvolutionMetrics:
    def __init__(self, metrics=None, multipliers=None, qbadic_bases=None, qbadic_norm='l1', 
                 track_matrices=True, fdc_correlation_type='pearson', save_dir='metrics'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.metrics = metrics if metrics is not None else ['euclidean', 'qbadic']
        self.multipliers = multipliers if multipliers is not None else [10]
        self.qbadic_bases = qbadic_bases if qbadic_bases is not None else [2]
        self.qbadic_norm = qbadic_norm
        self.track_matrices = track_matrices
        self.fdc_correlation_type = fdc_correlation_type

        # Construct configurations to track
        self.tracked_configs = []
        for metric in self.metrics:
            if metric == 'qbadic':
                for base in self.qbadic_bases:
                    for mult in self.multipliers:
                        name = f'qbadic_b{base}_m{mult}'
                        self.tracked_configs.append({
                            'name': name, 'metric': 'qbadic', 
                            'base': base, 'multiplier': mult, 'qbadic_norm': self.qbadic_norm
                        })
            elif metric == 'padic':
                for base in self.qbadic_bases:
                    name = f'padic_b{base}'
                    self.tracked_configs.append({
                        'name': name, 'metric': 'padic', 
                        'base': base, 'multiplier': None, 'qbadic_norm': None
                    })
            else:
                self.tracked_configs.append({
                    'name': metric, 'metric': metric, 
                    'base': None, 'multiplier': None, 'qbadic_norm': None
                })
        
        # Initialize History
        self.history = {
            'generation': [],
            'best_fitness': [],
            'mean_fitness': [],
            'worst_fitness': [],
            'fdc_sample_size': [],
        }
        
        # Add dynamic keys
        for config in self.tracked_configs:
            self.history[f"diversity_{config['name']}"] = []
            self.history[f"correlation_{config['name']}"] = []
        
        self.SNAPSHOT_GENS = [0, 100, 500, 1000]

    def record(self, generation, population, is_final=False):
        """
        Record metrics based on the generation tier.
        """
        from nn_reals.Population import Population
        
        # --- Tier 1: Cheap Scalars (Every Generation) ---
        fitnesses = population.get_fitnesses()
        self.history['generation'].append(generation)
        self.history['best_fitness'].append(float(np.min(fitnesses)))
        self.history['mean_fitness'].append(float(np.mean(fitnesses)))
        self.history['worst_fitness'].append(float(np.max(fitnesses)))
        
        # --- Tier 2: Trend Interval (Every 10 Generations OR Final) ---
        if generation % 10 == 0 or is_final:
            for config in self.tracked_configs:
                # Diversity
                div = population.population_diversity(
                    metric=config['metric'], 
                    base=config['base'], 
                    multiplier=config['multiplier'], 
                    qbadic_norm=config['qbadic_norm'], 
                    n_samples=100
                )
                self.history[f"diversity_{config['name']}"].append(div['mean_distance'])
            
            # Fitness-Distance Correlation (sampled)
            self._record_correlation(population)
        else:
            # Keep lists aligned
            self.history['fdc_sample_size'].append(None)
            for config in self.tracked_configs:
                self.history[f"diversity_{config['name']}"].append(None)
                self.history[f"correlation_{config['name']}"].append(None)

        # --- Tier 3: Snapshots (Specific Generations) ---
        if (generation in self.SNAPSHOT_GENS or is_final) and self.track_matrices:
            print(f" >> Creating Snapshot for Generation {generation}...")
            
            snapshot_data = {
                'generation': generation,
                'fitnesses': fitnesses
            }
            
            # Compute Full NxN Matrices for all tracked metrics
            for config in self.tracked_configs:
                matrix = population.all_pairwise_distances(
                    metric=config['metric'], 
                    base=config['base'], 
                    multiplier=config['multiplier'], 
                    qbadic_norm=config['qbadic_norm']
                )
                snapshot_data[f"matrix_{config['name']}"] = matrix
            
            # Save Compressed
            filename = self.save_dir / f'snapshot_gen_{generation}.npz'
            np.savez_compressed(filename, **snapshot_data)
            print(f" >> Snapshot saved: {filename}")

    def _record_correlation(self, population):
        """Helper to calculate fitness-distance correlation on a subset."""
        from nn_reals.Population import Population
        
        best_net = population.get_best_networks(n=1)
        best_fitness = best_net.fitness()
        
        # Sample subset
        sample_size = min(50, population.pop_size)
        indices = np.random.choice(population.pop_size, sample_size, replace=False)
        
        fitness_diffs = []
        # Prepare lists for all configs
        dist_lists = {config['name']: [] for config in self.tracked_configs}
        
        for idx in indices:
            net = population[idx]
            if net is best_net:
                continue
            
            fitness_diffs.append(abs(net.fitness() - best_fitness))
            
            for config in self.tracked_configs:
                d = Population.genetic_distance(
                    best_net, net, 
                    config['metric'], config['base'], config['multiplier'], config['qbadic_norm']
                )
                dist_lists[config['name']].append(d)
        
        # Compute Correlation
        def get_corr(dist_array):
            if len(dist_array) < 2: return 0.0
            if len(set(dist_array)) < 2: return 0.0 
            if len(set(fitness_diffs)) < 2: return 0.0
            
            if self.fdc_correlation_type == 'spearman':
                val, _ = spearmanr(fitness_diffs, dist_array)
            elif self.fdc_correlation_type == 'kendall':
                val, _ = kendalltau(fitness_diffs, dist_array)
            else: # Default strict Pearson
                val, _ = pearsonr(fitness_diffs, dist_array)
                
            return 0.0 if np.isnan(val) else val

        for config in self.tracked_configs:
            corr = get_corr(dist_lists[config['name']])
            self.history[f"correlation_{config['name']}"].append(corr)

        self.history['fdc_sample_size'].append(len(indices))

    def get_results_summary(self):
        """
        Calculate and return summary statistics: 
        Start/End diversity, Middle 50% Avg Diversity, Avg FDC.
        """
        summary = {}
        generations = self.history['generation']
        if not generations:
            return summary
            
        max_gen = generations[-1]
        mid_start = max_gen * 0.25
        mid_end = max_gen * 0.75
        
        mid_indices = [i for i, g in enumerate(generations) if mid_start <= g <= mid_end]
        
        for config in self.tracked_configs:
            name = config['name']
            
            # Diversity
            div_key = f"diversity_{name}"
            div_vals = self.history[div_key]
            
            # Filter None
            valid_divs = [(i, v) for i, v in enumerate(div_vals) if v is not None]
            
            if not valid_divs:
                continue
                
            start_div = valid_divs[0][1]
            end_div = valid_divs[-1][1]
            
            # Middle 50%
            mid_vals = [div_vals[i] for i in mid_indices if div_vals[i] is not None]
            mid_avg = np.mean(mid_vals) if mid_vals else 0.0
            
            # FDC
            corr_key = f"correlation_{name}"
            corr_vals = [v for v in self.history[corr_key] if v is not None]
            avg_fdc = np.mean(corr_vals) if corr_vals else 0.0
            
            # Avg Sample Size
            sample_size_key = "fdc_sample_size"
            sample_size_vals = [v for v in self.history.get(sample_size_key, []) if v is not None]
            avg_sample_size = np.mean(sample_size_vals) if sample_size_vals else 0.0

            summary[name] = {
                'start_diversity': start_div,
                'end_diversity': end_div,
                'mid_50_diversity': mid_avg,
                'avg_fdc': avg_fdc,
                'avg_fdc_sample_size': avg_sample_size
            }
            
        return summary

    def calculate_orthogonality(self, population, metric_pairs=None, correlation_type='pearson', n_samples=100):
        """
        Calculate correlation between distance matrices of different metrics.
        Higher correlation = metrics provide similar information (low orthogonality).
        Lower correlation = metrics track different things (high orthogonality).
        
        metric_pairs: List of tuples. Can be (config_dict, config_dict) or ('name1', 'name2').
                      If None, compares the first tracked metric vs all others.
        """
        from nn_reals.Population import Population
        
        # Helper to find config by name
        def get_config(item):
            if isinstance(item, dict): return item
            if isinstance(item, str):
                for c in self.tracked_configs:
                    if c['name'] == item: return c
                raise ValueError(f"Metric name '{item}' not found in tracked configs.")
            raise ValueError(f"Invalid metric identifier: {item}")

        # If no pairs specified, compare first metric (usually Euclidean) vs all others
        if metric_pairs is None:
            metric_pairs = []
            base_config = self.tracked_configs[0] # Usually Euclidean
            for i in range(1, len(self.tracked_configs)):
                metric_pairs.append((base_config, self.tracked_configs[i]))
        else:
            # Resolve strings to configs
            metric_pairs = [(get_config(p[0]), get_config(p[1])) for p in metric_pairs]
        
        results = {}
        
        # Pre-calculate distances for the sample to avoid re-sampling for each pair
        # We need a fixed set of pairs of individuals to compare distances on
        
        if population.pop_size < 2:
            return {}
            
        max_pairs = population.pop_size * (population.pop_size - 1) // 2
        n_samples = min(n_samples, max_pairs)
        
        # Generate fixed pairs of indices
        pair_indices = []
        for _ in range(n_samples):
             idx1, idx2 = np.random.choice(population.pop_size, 2, replace=False)
             pair_indices.append((idx1, idx2))
             
        # Cache distances for each config for these pairs
        # {config_name: [d1, d2, ...]}
        distance_cache = {}
        
        # Identify which configs we need
        needed_configs = set()
        for c1, c2 in metric_pairs:
            needed_configs.add(c1['name'])
            needed_configs.add(c2['name'])
            
        # Compute distances
        for config in self.tracked_configs:
            if config['name'] not in needed_configs:
                continue
                
            dists = []
            for idx1, idx2 in pair_indices:
                d = Population.genetic_distance(
                    population[idx1], population[idx2],
                    config['metric'], config['base'], config['multiplier'], config['qbadic_norm']
                )
                dists.append(d)
            distance_cache[config['name']] = dists
            
        # Compute correlations
        for c1, c2 in metric_pairs:
            d1 = distance_cache[c1['name']]
            d2 = distance_cache[c2['name']]
            
            if len(d1) < 2 or len(set(d1)) < 2 or len(set(d2)) < 2:
                val = 0.0
            elif correlation_type == 'spearman':
                val, _ = spearmanr(d1, d2)
            elif correlation_type == 'kendall':
                val, _ = kendalltau(d1, d2)
            else:
                val, _ = pearsonr(d1, d2)
                
            results[f"{c1['name']} vs {c2['name']}"] = val
            
        return results

    def save(self):
        """Save the lightweight history to JSON."""
        filepath = self.save_dir / 'metrics_history.json'
        
        # Convert all numpy floats to python floats for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2, default=convert)
        print(f"Metrics history saved to {filepath}")