import numpy as np
from fractions import Fraction
from nn_reals.Network import Network

class Population:
    '''Class representing a population of neural networks for neuroevolution.'''
    def __init__(self, X, Y, layers=None, task='regression', pop_size=10):
        self.X = X
        self.Y = Y
        self.layers = layers
        self.task = task
        self.pop_size = pop_size
        self.pop = self.initialize_population(self.X, self.Y, self.layers, self.task, self.pop_size)
        
    # Each individual in the population is a network with random weights and bias
    def initialize_population(self, X, Y, layers, task, pop_size):
        return [Network(X, Y, layers=layers, task=task) for _ in range(pop_size)]
    
    # Genetic distance between two networks based on their genomes (p, multiplier, and padic_norm are only used for p-adic metric and qbadic)
    @staticmethod
    def genetic_distance(net1, net2, metric, base, multiplier, qbadic_norm):
        genome_diff = net1.genome - net2.genome
        
        if metric == 'euclidean':
            return np.sqrt(np.sum(genome_diff ** 2))
        elif metric == 'manhattan':
            return np.sum(np.abs(genome_diff))
        elif metric == 'chebyshev':
            return np.max(np.abs(genome_diff))
        elif metric == 'qbadic':    # Quantized p-adic distance
            if qbadic_norm == 'linf':
                return Population.qbadic_distance_linf(genome_diff, base, multiplier)
            elif qbadic_norm == 'l1':
                return Population.qbadic_distance_l1(net1.genome, net2.genome, base, multiplier)
            elif qbadic_norm == 'l2':
                return Population.qbadic_distance_l2(net1.genome, net2.genome, base, multiplier)
            else:
                raise ValueError(f"Unknown p-adic norm: {qbadic_norm}")
        elif metric == 'padic':
            return Population.padic_distance_l1(genome_diff, base)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    # quantization map
    # Round() for rounding half to even, int() for floor, np.cell() for ceiling (try Stochastic rounding?)
    @staticmethod
    def q_map(x, multiplier):
        return round(x * multiplier)

    # Compute pseudo b-adic valuation for floats by scaling to integers.
    @staticmethod
    def qbadic_valuation(x, y, base, multiplier):
        """
        Compute pseudo b-adic valuation for floats by scaling to integers.
        Vectorized for numpy arrays.
        """
        # Ensure inputs are arrays
        x = np.asarray(x)
        y = np.asarray(y)
        
        # Quantize the inputs
        q_x = np.round(x * multiplier)
        q_y = np.round(y * multiplier)
        q_diff = np.abs(q_x - q_y).astype(int)
        
        # Handle zeros -> inf valuation
        valuations = np.zeros_like(q_diff, dtype=float)
        zero_mask = (q_diff == 0)
        valuations[zero_mask] = float('inf')
        
        # Calculate valuation for non-zeros
        # vectorizing the "while divisible by base" logic
        # We can use iterative division on the mask of non-zeros
        non_zero_mask = ~zero_mask
        temp_diff = q_diff[non_zero_mask]
        
        counts = np.zeros_like(temp_diff, dtype=int)
        
        if temp_diff.size > 0:
            while True:
                is_divisible = (temp_diff % base == 0)
                if not np.any(is_divisible):
                    break
                temp_diff[is_divisible] //= base
                counts[is_divisible] += 1
                
        valuations[non_zero_mask] = counts
        return valuations

    @staticmethod
    def qbadic_norm_component(x, y, base, multiplier):
        """Vectorized p-adic norm component."""
        vals = Population.qbadic_valuation(x, y, base, multiplier)
        # norm = base^(-val)
        # handle inf
        norms = np.zeros_like(vals)
        finite_mask = (vals != float('inf'))
        norms[finite_mask] = list(base) ** (-vals[finite_mask]) if isinstance(base, list) else base ** (-vals[finite_mask])
        return norms

    @staticmethod
    def qbadic_distance_linf(vector, base, multiplier):
        """Vectorized Linf p-adic distance."""
        # vector is already difference array usually, or we treat it as such against 0 vector
        # check if input is difference or raw vector (usually raw difference in context)
        # implementation assumes 'vector' is the difference genome_diff
        
        # We need to compute norm of each component against 0
        norms = Population.qbadic_norm_component(vector, np.zeros_like(vector), base, multiplier)
        
        # Filter out negligible components (already handled by diff==0 -> norm=0 logic mostly, 
        # but original code had abs(x) > 1e-10 check. 
        # The quantized valuation handles small diffs by rounding them to 0 if they are small enough given the multiplier.
        # But let's respect the original tolerance for safety if needed, though quantization usually overrides it.
        # If quantized diff is 0, norm is 0.
        
        return np.max(norms) if norms.size > 0 else 0.0
    
    @staticmethod
    def qbadic_distance_l1(net1, net2, base, multiplier):
        """Vectorized L1 p-adic distance."""
        diff = net1 - net2 # Assuming net1, net2 are genomes (numpy arrays)
        # Original code used net1[i] - net2[i] and passed 0.0 as second arg to scalar func.
        # Equivalent to passing diff and 0s.
        norms = Population.qbadic_norm_component(diff, np.zeros_like(diff), base, multiplier)
        return np.sum(norms)
    
    @staticmethod
    def qbadic_distance_l2(net1, net2, base, multiplier):
        """Vectorized L2 p-adic distance."""
        diff = net1 - net2
        norms = Population.qbadic_norm_component(diff, np.zeros_like(diff), base, multiplier)
        return np.sqrt(np.sum(norms ** 2))
    
    # Exact p-adic valuation for rationals using Fraction (no correlation found)
    @staticmethod
    def padic_valuation(x, p):
        if x == 0:
            return float('inf')
        
        # Convert float to exact rational
        frac = Fraction(x).limit_denominator()
        
        # Compute ν_p(numerator) - ν_p(denominator)
        def count_factors(n, p):
            if n == 0:
                return float('inf')
            n = abs(n)
            count = 0
            while n % p == 0:
                n //= p
                count += 1
            return count
        
        return count_factors(frac.numerator, p) - count_factors(frac.denominator, p)

    @staticmethod
    def padic_norm_component(x, p):
        val = Population.padic_valuation(x, p)
        if val == float('inf'):
            return 0.0
        return p ** (-val)

    @staticmethod
    def padic_distance_l1(vector, p):
        total = 0.0
        for x in vector:
            if abs(x) > 1e-10:
                total += Population.padic_norm_component(x, p)
        return total

    # Calculates the average distance between pairs out of n randomly chosen individuals
    def population_diversity(self, metric, base, multiplier, qbadic_norm, n_samples):
        if self.pop_size < 2:
            raise ValueError("Population must have at least 2 networks")
        
        # Limit samples to avoid exceeding max possible pairs
        max_possible_pairs = self.pop_size * (self.pop_size - 1) // 2 # All possible pairs
        n_samples = min(n_samples, max_possible_pairs)
        
        distances = []

        # Randomly sample pairs and compute their distances
        for _ in range(n_samples):
            # Select two different random networks
            idx1, idx2 = np.random.choice(self.pop_size, 2, replace=False)
            net1, net2 = self.pop[idx1], self.pop[idx2]
            
            # Compute distance
            dist = Population.genetic_distance(net1, net2, metric, base, multiplier, qbadic_norm)
            distances.append(dist)
        
        distances = np.array(distances)
        
        return {
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances)
        }
    
    def get_fitnesses(self):
        '''Returns an array of fitness values for all networks in the population.'''
        return np.array([net.fitness() for net in self.pop])
    
    # returns the n best networks in the population
    def get_best_networks(self, n=1):
        sorted_pop = sorted(self.pop, key=lambda net: net.fitness())
        if n == 1:
            return sorted_pop[0]
        return sorted_pop[:n]
    
    # returns a distance matrix
    def all_pairwise_distances(self, metric, base, multiplier, qbadic_norm):
        n = self.pop_size
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = Population.genetic_distance(self.pop[i], self.pop[j], metric, base, multiplier, qbadic_norm)
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances

    def __len__(self):
        return self.pop_size
    
    def __getitem__(self, index):
        return self.pop[index]