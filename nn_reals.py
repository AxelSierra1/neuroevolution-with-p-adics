from sage.all import *
import numpy as np

np.random.seed(42)

# Problem examples: (X: features, y: labels) 
# XOR problem 
x_XOR = np.array([[0,0], [0,1], [1,0], [1,1]])
y_XOR = np.array([[0], [1], [1], [0]])

# AND problem
x_AND = np.array([[0,0], [0,1], [1,0], [1,1]])
y_AND = np.array([[0], [0], [0], [1]])

# Function approximation problem
# Define input range
x = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1)  # 100 points between -π and π
# Define output function
y = 0.5 * np.cos(2 * x)

# Neural network with arbitrary layer sizes
# When specifying layers, it should include the ouput layer always
class Network:
    def __init__(self, X, Y, layers=None, genome=None):
        self.X = X # Features
        self.Y = Y # Labels
        
        # Default to single layer (original behavior) if layers not specified
        if layers is None:
            layers = [Y.shape[1]] # One layer matching the number of outputs (number of outputs = Y.shape[1])
        self.layers = [X.shape[1]] + layers  # List containing the input layers and additional layers (X.shape[1] = number of input features)
        
        # Genome is a vector containing all genome information (weights then biases
        if genome is None:
            # initialize random values for weights and 0 for the biases in the genome
            self.genome = self._initialize_genome()
        else:
            self.genome = genome.copy()

    def _initialize_genome(self):
        genome_parts = []
        
        # Weights
        for i in range(len(self.layers) - 1):
            size = self.layers[i] * self.layers[i+1]
            genome_parts.append(np.random.randn(size))
        
        # Biases
        for i in range(len(self.layers) - 1):
            size = self.layers[i+1]
            genome_parts.append(np.zeros(size))
        
        return np.concatenate(genome_parts)

    # Decodes a flat genome array into weights and biases matrices/vectors
    def decode_genome(self):
        # Keeps track of where we are in the flat genome
        idx = 0

        weights = []
        # Decode weights
        for i in range(len(self.layers) - 1):
            rows, cols = self.layers[i], self.layers[i + 1]
            size = rows * cols
            weight_matrix = self.genome[idx:idx + size].reshape(rows, cols)
            weights.append(weight_matrix)
            idx += size
        
        biases = []
        # Decode biases
        for i in range(len(self.layers) - 1):
            size = self.layers[i + 1]
            bias_vector = self.genome[idx:idx + size]
            biases.append(bias_vector)
            idx += size
        
        return weights, biases
    
    def print_genome(self):
        print(self.genome)

    def activation(self, input):
        return  1 / (1 + np.exp(-input))

    '''
    This implementation treats inputs features as row vectors instead of column vectors which results in a row vector output:
    [x1 x2 x3]   [w11 w12]   [b1]
               * [w21 w22] + [b2] = [x1' x2']
                 [w31 w32]   [b3]

    More generally, it does this for every possible data input to get the full output at once:
    [x1 x2 x3]   [w11 w12]   [b1]   [x1' x2']
    [y1 y2 y3] * [w21 w22] + [b2] = [y1' y2']
    [z1 z2 z3]   [w31 w32]   [b3]   [z1' z2']

    (This is sometimes done the other way around, simply taking the transpose of both sides:
    (xA + b)T = (x')T -> (xA)T + bT=x'T -> ATxT + bT=x'T)
    '''
    def output(self):
        weights, biases = self.decode_genome()
        current_input = self.X
        for i in range(len(weights)):
            current_input = self.activation(np.dot(current_input, weights[i]) + biases[i]) # Computes the layer operations in a loop until it reaches the output layer
        return current_input # Returns a matrix containing the outputs of all possible entries (all our data) (each row corresponds to a data point)
    
    # Function to determine how good the network is
    # Fitness for regression should be mse, for classification should be cross-entropy
    # Generally, higher value is better, but in this implementation, lower is better (using fitness as an error function)
    def fitness(self):
        return np.mean((self.Y - self.output()) ** 2)
    
    # Adds perturbations/noise to the weights and biases
    # rate tells us how strong the noise is
    def mutate(self, rate=0.1, prob=0.1):
        noise = np.random.randn(self.genome.shape[0]) * rate
        mask = np.random.rand(self.genome.shape[0]) < prob
        
        self.genome += noise * mask
    
    # Each individual in the population is a network with random weights and bias:
    @staticmethod
    def initialize_population(size, X, Y, layers=None):
        return [Network(X, Y, layers) for _ in range(size)]

    # We choose 2 parents from a cluster of k candidates chosen at random (we choose the 2 with best fitness in this cluster)
    @staticmethod
    def select_parents(pop, k=3):
        # First parent
        candidates1 = np.random.choice(pop, k, replace=False)
        parent1 = min(candidates1, key=lambda net: net.fitness())

        # Second parent, ensure its different than parent1
        while True:
            candidates2 = np.random.choice(pop, k, replace=False)
            parent2 = min(candidates2, key=lambda net: net.fitness())
            if parent2 is not parent1:
                break

        return parent1, parent2

    # Simply average all weight matrices and bias vectors of both parents
    @staticmethod
    def crossover_average(parent1, parent2):
        child_genome = (parent1.genome + parent2.genome) / 2
        
        return Network(parent1.X, parent1.Y, layers=parent1.layers[1:], genome=child_genome)
    
    # Chooses randomly a point in the list of all parameters of the child, parameters before that point come from parent1, after that point come from parent2
    # Can choose the amount of points to split the genes from both parents
    @staticmethod
    def crossover_npoints(parent1, parent2, n_points=1):
        genome_length = len(parent1.genome)
        
        if n_points == 1:
            # Single-point crossover
            crossover_point = np.random.randint(1, genome_length)
            child_genome = np.concatenate([
                parent1.genome[:crossover_point],
                parent2.genome[crossover_point:]
            ])
        else:
            # Multi-point crossover
            crossover_points = sorted(np.random.choice(
                range(1, genome_length), 
                size=min(n_points, genome_length - 1), 
                replace=False
            ))
            crossover_points = [0] + crossover_points + [genome_length]
            
            # Create child by alternating between parents
            child_genome = np.zeros_like(parent1.genome)
            for i in range(len(crossover_points) - 1):
                start, end = crossover_points[i], crossover_points[i + 1]
                if i % 2 == 0:
                    child_genome[start:end] = parent1.genome[start:end]
                else:
                    child_genome[start:end] = parent2.genome[start:end]
        
        return Network(parent1.X, parent1.Y, layers=parent1.layers[1:], genome=child_genome)
    
    # Each parameter chosen randomly from either parent, prob is probability of choosing a parameter from parent1
    @staticmethod
    def crossover_uniform(parent1, parent2, prob=0.5):
        # Create uniform crossover mask
        mask = np.random.random(parent1.genome.shape) < prob
        
        # If mask == True, use parent1; if False, use parent2
        child_genome = np.where(mask, parent1.genome, parent2.genome)
        
        return Network(parent1.X, parent1.Y, layers=parent1.layers[1:], genome=child_genome)
    
    # stagnation_count is the number of generations without improvement
    @staticmethod
    def adaptive_mutation_rate(base_rate, fitness_improvement, stagnation_count, min_rate=0.01, max_rate=0.6):
        
        # If fitness is improving, decrease mutation rate
        if fitness_improvement > 1e-6:
            rate = base_rate * 0.9
        # If stagnating, increase mutation rate
        elif stagnation_count > 5:
            rate = base_rate * (1.0 + 0.1 * min(stagnation_count - 5, 10))
        else:
            rate = base_rate
        
        return np.clip(rate, min_rate, max_rate)
    
    @staticmethod
    def evolution(X, Y, layers=None, generations=100, pop_size=20, k=3, mutation_rate=0.1, 
                  mutation_prob=0.1, elitism_rate=0.05, crossover_method='average',
                  crossover_kwargs=None, adaptive_mutation=True, early_stopping=None):
        
        if crossover_kwargs is None:
            crossover_kwargs = {}
        
        population = Network.initialize_population(pop_size, X, Y, layers)
        
        prev_best_fitness = float('inf')
        stagnation_count = 0
        current_mutation_rate = mutation_rate
        
        for gen in range(generations):
            population.sort(key=lambda net: net.fitness())
            
            current_best_fitness = population[0].fitness()
            fitness_improvement = prev_best_fitness - current_best_fitness
            
            if adaptive_mutation:
                if fitness_improvement <= 1e-6:
                    stagnation_count += 1
                else:
                    stagnation_count = 0
                
                current_mutation_rate = Network.adaptive_mutation_rate(
                    mutation_rate, fitness_improvement, stagnation_count
                )
            
            elitism = max(1, int(elitism_rate * pop_size))
            new_population = population[:elitism]
            
            while len(new_population) < pop_size:
                p1, p2 = Network.select_parents(population, k)
                
                if crossover_method == 'average':
                    child = Network.crossover_average(p1, p2)
                elif crossover_method == 'point':
                    n_points = crossover_kwargs.get('n_points', 1)
                    child = Network.crossover_npoints(p1, p2, n_points=n_points)
                elif crossover_method == 'uniform':
                    prob = crossover_kwargs.get('prob', 0.5)
                    child = Network.crossover_uniform(p1, p2, prob=prob)
                else:
                    raise ValueError(f"Unknown crossover method: {crossover_method}")
                
                child.mutate(rate=current_mutation_rate, prob=mutation_prob)
                new_population.append(child)
            
            population = new_population
            best_net = population[0]
            best_fitness = best_net.fitness()
            
            print(f"Generation {gen+1}, best fitness: {best_fitness:.6f}, "
                  f"mutation rate: {current_mutation_rate:.4f}, "
                  f"stagnation: {stagnation_count}")
            
            prev_best_fitness = current_best_fitness
            
            if best_fitness < 1e-3:
                print("Converged!")
                break
            if early_stopping is not None and stagnation_count >= early_stopping:
                print(f"Early stopping at generation {gen+1}")
                break
        
        return population[0]
    
    # Calculate the distance between two networks based on their genomes
    @staticmethod
    def distance(net1, net2, metric='euclidean'):
        if len(net1.genome) != len(net2.genome):
            raise ValueError("Networks must have the same architecture to compute distance")
        
        genome_diff = net1.genome - net2.genome
        
        if metric == 'euclidean':
            # Euclidean distance: sqrt(sum of squared differences)
            return np.sqrt(np.sum(genome_diff ** 2))
        elif metric == 'manhattan':
            # Manhattan distance: sum of absolute differences
            return np.sum(np.abs(genome_diff))
        elif metric == 'chebyshev':
            # Chebyshev distance: max absolute difference
            return np.max(np.abs(genome_diff))
        else:
            raise ValueError(f"Unknown metric: {metric}")


best_net_avg = Network.evolution(x, y, layers=[2, 2, 1], generations=1000, pop_size=1000, k=10, mutation_rate=0.15, elitism_rate=0.01, 
                                crossover_method='average', adaptive_mutation=True)

best_net_multi = Network.evolution(x, y, layers=[2, 2, 1], generations=1000, pop_size=1000, k=10, mutation_rate=0.15, elitism_rate=0.01, 
                                  crossover_method='point', crossover_kwargs={'n_points': 1}, adaptive_mutation=True)

best_net_uniform = Network.evolution(x, y, layers=[2, 2, 1], generations=1000, pop_size=1000, k=10, mutation_rate=0.15, elitism_rate=0.01, 
                                    crossover_method='uniform', crossover_kwargs={'prob': 0.6}, adaptive_mutation=True)

print(f"\nFinal Results:")
print(f"Average crossover fitness: {best_net_avg.fitness()}")
print(f"Multi-point crossover fitness: {best_net_multi.fitness()}")
print(f"Uniform crossover fitness: {best_net_uniform.fitness()}")