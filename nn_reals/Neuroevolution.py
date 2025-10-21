import numpy as np
from nn_reals.Network import Network
from nn_reals.EvolutionMetrics import EvolutionMetrics

class Neuroevolution:
    '''Optimized neuroevolution with selection, crossover, and mutation.'''
    
    def __init__(self, population):
        self.population = population
        self.pop_history = [self.population]
        self._fitness_cache = {}  # Cache fitness calculations

    def _get_fitness(self, net):
        """Cache fitness values to avoid recomputation"""
        net_id = id(net)
        if net_id not in self._fitness_cache:
            self._fitness_cache[net_id] = net.fitness()
        return self._fitness_cache[net_id]

    def select_parents(self, k):
        """Tournament selection - vectorized for speed"""
        # Select all candidates at once
        candidates = np.random.choice(self.population.pop, k * 2, replace=False)
        
        # Get fitness for all candidates (use cached values)
        fitness_vals = np.array([self._get_fitness(net) for net in candidates])
        
        # Get indices of two best from separate tournaments
        idx1 = np.argmin(fitness_vals[:k])
        idx2 = k + np.argmin(fitness_vals[k:])
        
        return candidates[idx1], candidates[idx2]

    def crossover(self, p1, p2, method='average', **kwargs):
        """Unified crossover method - more compact"""
        if method == 'average':
            child_genome = (p1.genome + p2.genome) * 0.5
        
        elif method == 'point':
            n_points = kwargs.get('n_points', 1)
            genome_len = len(p1.genome)
            
            if n_points == 1:
                point = np.random.randint(1, genome_len)
                child_genome = np.concatenate([p1.genome[:point], p2.genome[point:]])
            else:
                points = np.sort(np.random.choice(genome_len - 1, min(n_points, genome_len - 1), replace=False) + 1)
                points = np.concatenate([[0], points, [genome_len]])
                
                # Vectorized alternating selection
                child_genome = np.empty_like(p1.genome)
                for i in range(len(points) - 1):
                    parent = p1 if i % 2 == 0 else p2
                    child_genome[points[i]:points[i+1]] = parent.genome[points[i]:points[i+1]]
        
        elif method == 'uniform':
            mask = np.random.random(p1.genome.shape) < kwargs.get('prob', 0.5)
            child_genome = np.where(mask, p1.genome, p2.genome)
        
        else:
            raise ValueError(f"Unknown crossover method: {method}")
        
        return Network(p1.X, p1.Y, layers=p1.layers[1:], genome=child_genome, task=p1.task)

    @staticmethod
    def adaptive_mutation_rate(base_rate, fitness_improvement, stagnation_count, 
                               min_rate=0.01, max_rate=0.6):
        """Adaptive mutation rate based on progress"""
        if fitness_improvement > 1e-6:
            rate = base_rate * 0.9
        elif stagnation_count > 5:
            rate = base_rate * (1.0 + 0.1 * min(stagnation_count - 5, 10))
        else:
            rate = base_rate
        
        return np.clip(rate, min_rate, max_rate)
    
    def evolution(self, generations=100, k=3, mutation_rate=0.15, mutation_prob=0.15, 
                  elitism_rate=0.05, crossover_method='average', crossover_kwargs=None, 
                  track_metrics=True, adaptive_mutation=True, early_stopping=None, 
                  task='regression', verbose=True):
        
        crossover_kwargs = crossover_kwargs or {}
        metrics = EvolutionMetrics(
            save_dir='metrics', 
            metrics=['euclidean', 'manhattan', 'chebyshev', 'padic', 'qpadic'], 
            multipliers=[1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 10000],
            qpadic_primes=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 36, 100, 1000]
        ) if track_metrics else None
        
        prev_best_fitness = float('inf')
        stagnation_count = 0
        current_mutation_rate = mutation_rate
        elitism = max(1, int(elitism_rate * self.population.pop_size))
        
        for gen in range(generations):
            # Clear fitness cache at start of generation
            self._fitness_cache.clear()
            
            # Sort population once with cached fitness
            self.population.pop.sort(key=self._get_fitness)
            
            if metrics:
                metrics.record_generation(gen, self.population, qpadic_norm='l1')
            
            current_best_fitness = self._get_fitness(self.population.pop[0])
            fitness_improvement = prev_best_fitness - current_best_fitness
            
            # Adaptive mutation
            if adaptive_mutation:
                stagnation_count = stagnation_count + 1 if fitness_improvement <= 1e-6 else 0
                current_mutation_rate = self.adaptive_mutation_rate(
                    mutation_rate, fitness_improvement, stagnation_count
                )
            
            # Elitism + offspring generation
            new_population = self.population.pop[:elitism]
            
            # Batch create offspring
            offspring_needed = self.population.pop_size - elitism
            for _ in range(offspring_needed):
                p1, p2 = self.select_parents(k)
                child = self.crossover(p1, p2, method=crossover_method, **crossover_kwargs)
                child.mutate(rate=current_mutation_rate, prob=mutation_prob)
                new_population.append(child)
            
            self.population.pop = new_population
            self.pop_history.append(self.population.pop[:])
            
            if verbose:
                print(f"Gen {gen+1}: fitness={current_best_fitness:.6f}, "
                      f"mut_rate={current_mutation_rate:.4f}, stag={stagnation_count}")
            
            prev_best_fitness = current_best_fitness
            
            # Early stopping
            if current_best_fitness < 1e-4:
                print("Converged!")
                break
            if early_stopping and stagnation_count >= early_stopping:
                print(f"Early stopping at generation {gen+1}")
                break
        
        if metrics:
            metrics.save(filename=f'run_{generations}gen.json')
            metrics.summary_report()
        
        return self.population.pop[0]