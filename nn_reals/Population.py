import numpy as np

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
    
    # Genetic distance between two networks based on their genomes (p, presicion, and padic_norm are only used for p-adic metric)
    @staticmethod
    def genetic_distance(net1, net2, metric, p=11, precision=3, padic_norm='l1'):
        genome_diff = net1.genome - net2.genome
        
        if metric == 'euclidean':
            return np.sqrt(np.sum(genome_diff ** 2))
        elif metric == 'manhattan':
            return np.sum(np.abs(genome_diff))
        elif metric == 'chebyshev':
            return np.max(np.abs(genome_diff))
        elif metric == 'padic':
            if padic_norm == 'linf':
                return Population.padic_distance_linf(genome_diff, p, precision)
            elif padic_norm == 'l1':
                return Population.padic_distance_l1(genome_diff, p, precision)
            elif padic_norm == 'l2':
                return Population.padic_distance_l2(genome_diff, p, precision)
            else:
                raise ValueError(f"Unknown p-adic norm: {padic_norm}")
        else:
            raise ValueError(f"Unknown metric: {metric}")

    # Compute p-adic valuation for floats by scaling to integers.
    @staticmethod
    def padic_valuation(x, p, precision=3):
        if x == 0:
            return float('inf')
        # Scale float to integer
        x_scaled = round(abs(x) * precision)
        
        if x_scaled == 0:  # Very small values round to zero
            return float('inf')
        
        count = 0
        while x_scaled % p == 0:
            x_scaled //= p
            count += 1
        return count

    # Compute p-adic norm |x|_p for a vector for a single component |x|_p = p^(-ν_p(x))
    @staticmethod
    def padic_norm_component(x, p, precision):
        val = Population.padic_valuation(x, p, precision)
        if val == float('inf'):
            return 0.0
        return p ** (-val)

    # Linfinity p-adic distance: ||v||_p,inf = max_i |v_i|_p, This is the "ultrametric" approach. distance is determined by the single component with largest p-adic norm.
    @staticmethod
    def padic_distance_linf(vector, p, precision):
        norms = [Population.padic_norm_component(x, p, precision) for x in vector if abs(x) > 1e-10]
        if not norms:  # Zero vector
            return 0.0
        return max(norms)
    
    # L1 p-adic norm. Considers the total accumulated p-adic difference across the genome
    @staticmethod
    def padic_distance_l1(vector, p, precision):
        total = 0.0
        for x in vector:
            if abs(x) > 1e-10:
                total += Population.padic_norm_component(x, p, precision)
        return total
    
    # L2 p-adic norm. p-adic analogue of Euclidean distance. Weights larger p-adic differences more heavily than L1
    @staticmethod
    def padic_distance_l2(vector, p, precision):
        sum_squares = 0.0
        for x in vector:
            if abs(x) > 1e-10:
                norm = Population.padic_norm_component(x, p, precision)
                sum_squares += norm ** 2
        return np.sqrt(sum_squares)

    # Calculates the average distance between pairs out of n randomly chosen individuals
    def population_diversity(self, metric, n_samples=50):
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
            dist = Population.genetic_distance(net1, net2, metric=metric)
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
    def all_pairwise_distances(self, metric):
        n = self.pop_size
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = Population.genetic_distance(self.pop[i], self.pop[j], metric=metric)
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances

    def __len__(self):
        return self.pop_size
    
    def __getitem__(self, index):
        return self.pop[index]