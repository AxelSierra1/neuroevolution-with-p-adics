import numpy as np

from nn_reals.Network import Network

class Population:
    def __init__(self, X, Y, layers=None, task='regression', pop_size=100):
        self.X = X
        self.Y = Y
        self.layers = layers
        self.task = task
        self.pop_size = pop_size
        self.pop = self.initialize_population(self.X, self.Y, self.layers, self.task, self.pop_size)
        
    # Each individual in the population is a network with random weights and bias
    def initialize_population(self, X, Y, layers, task, pop_size):
        return [Network(X, Y, layers=layers, task=task) for _ in range(pop_size)]
    
    # Calculate the distance between two networks based on their genomes
    # Implement this notion of distance in:
    # Tracking how much networks evolve over generations
    # Detecting convergence
    # Analyzing speciation
    @staticmethod
    def distance(net1, net2, metric='euclidean'):
        if len(net1.genome) != len(net2.genome):
            raise ValueError("Networks must have the same architecture to compute distance")
        
        genome_diff = net1.genome - net2.genome
        
        if metric == 'euclidean': # Euclidean distance: sqrt(sum of squared differences)
            return np.sqrt(np.sum(genome_diff ** 2))
        elif metric == 'manhattan': # Manhattan distance: sum of absolute differences
            return np.sum(np.abs(genome_diff))
        elif metric == 'chebyshev': # Chebyshev distance: max absolute difference
            return np.max(np.abs(genome_diff))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    # Calculates the average distance between pairs out of n randomly chosen individuals
    def population_diversity(self, n_samples=50, metric='euclidean'):
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
            dist = Population.distance(net1, net2, metric=metric)
            distances.append(dist)
        
        distances = np.array(distances)
        
        return {
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances)
        }
    
    # return fitness list/array of all networks in population
    def get_fitnesses(self):
        return np.array([net.fitness() for net in self.pop])
    
    # returns the n best networks in the population
    def get_best_networks(self, n=1):
        sorted_pop = sorted(self.pop, key=lambda net: net.fitness())
        if n == 1:
            return sorted_pop[0]
        return sorted_pop[:n]
    
    # returns a distance matrix
    def all_pairwise_distances(self, metric='euclidean'):
        n = self.pop_size
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = Population.distance(self.pop[i], self.pop[j], metric=metric)
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances

    def __len__(self):
        return self.pop_size
    
    def __getitem__(self, index):
        return self.pop[index]