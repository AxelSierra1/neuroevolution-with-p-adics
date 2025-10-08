from sage.all import *
import numpy as np

np.random.seed(42)

# Problem examples: (X: features, y: labels) 
# XOR problem 
X_XOR = np.array([[0,0], [0,1], [1,0], [1,1]])
Y_XOR = np.array([[0], [1], [1], [0]])

# AND problem
X_AND = np.array([[0,0], [0,1], [1,0], [1,1]])
Y_AND = np.array([[0], [0], [0], [1]])

# Function approximation problem
# Define input range
X = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1)  # 100 points between -π and π
# Define output function
Y = 0.5 * np.cos(2 * X)

# Neural network with arbitrary layer sizes
# When specifying layers, it should include the ouput layer always
class network:
    def __init__(self, X, Y, layers=None, weights=None, biases=None):
        self.X = X # Features
        self.Y = Y # Labels
        
        # Default to single layer (original behavior) if layers not specified
        if layers is None:
            layers = [Y.shape[1]] # One layer matching the number of outputs (number of outputs = Y.shape[1])
        self.layers = [X.shape[1]] + layers  # List containing the input layers and additional layers (X.shape[1] = number of input features)
        
        if weights is None:
            self.weights = [] # List of weight matrices
            for i in range(len(self.layers) - 1): #Leave out the input layer
                self.weights.append(np.random.randn(self.layers[i], self.layers[i+1])) # Weight matrix for each layer: np.random.randn(rows, cols)
        else:
            self.weights = weights
            
        if biases is None:
            self.biases = []  # List of bias vectors
            for i in range(len(self.layers) - 1): #Leave out the input layer
                self.biases.append(np.zeros(self.layers[i+1])) # Bias vector for each layer: np.random.randn(length)
        else:
            self.biases = biases

        # Vector containing all genome information (weights then biases)
        self.genome = np.concatenate(
            [w.flatten() for w in self.weights] + [b.flatten() for b in self.biases]
        )


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
        current_input = self.X
        for i in range(len(self.weights)):
            current_input = self.activation(np.dot(current_input, self.weights[i]) + self.biases[i]) # Computes the layer operations in a loop until it reaches the output layer
        return current_input # Returns a matrix containing the outputs of all possible entries (all our data) (each row corresponds to a data point)
    
    # Function to determine how good the network is
    # Fitness for regression should be mse, for classification should be cross-entropy
    # Generally, higher value is better, but in this implementation, lower is better (using fitness as an error function)
    def fitness(self):
        return np.mean((self.Y - self.output()) ** 2)
    
    # Adds perturbations/noise to the weights and biases
    # rate tells us how strong the noise is
    def mutate(self, rate=0.1, prob=0.1):
        for i in range(len(self.weights)):
            noise_w = np.random.randn(*self.weights[i].shape) * rate
            noise_b = np.random.randn(*self.biases[i].shape) * rate

            # Mutation masks (boolean arrays)
            mask_w = np.random.rand(*self.weights[i].shape) < prob # This returns a boolean matrix
            mask_b = np.random.rand(*self.biases[i].shape) < prob

            # Apply the noise according to the ask ~0.1 of the weights are affected
            self.weights[i] += noise_w * mask_w
            self.biases[i] += noise_b * mask_b
    
    # Each individual in the population is a network with random weights and bias:
    @staticmethod
    def initialize_population(size, X, Y, layers=None):
        return [network(X, Y, layers) for _ in range(size)]

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
        child_weights = []
        child_biases = []
        
        for i in range(len(parent1.weights)):
            child_weights.append((parent1.weights[i] + parent2.weights[i]) / 2)
            child_biases.append((parent1.biases[i] + parent2.biases[i]) / 2)
        
        return network(parent1.X, parent1.Y, parent1.layers[1:], weights=child_weights, biases=child_biases)
    
    # Chooses randomly a point in the list of all parameters of the child, parameters before that point come from parent1, after that point come from parent2
    @staticmethod
    def crossover_single_point(parent1, parent2):
        child_weights = []
        child_biases = []
        
        # Calculate total number of parameters
        total_params = sum(w.size + b.size for w, b in zip(parent1.weights, parent1.biases))
        crossover_point = np.random.randint(1, total_params)
        
        current_param = 0
        for i in range(len(parent1.weights)):
            weight_size = parent1.weights[i].size
            bias_size = parent1.biases[i].size
            
            # Handle weights
            if current_param + weight_size <= crossover_point:
                # Take all from parent1
                child_weights.append(parent1.weights[i].copy())
            elif current_param >= crossover_point:
                # Take all from parent2
                child_weights.append(parent2.weights[i].copy())
            else:
                # Split within this weight matrix
                split_point = crossover_point - current_param
                flat_w1 = parent1.weights[i].flatten()
                flat_w2 = parent2.weights[i].flatten()
                child_flat = np.concatenate([flat_w1[:split_point], flat_w2[split_point:]])
                child_weights.append(child_flat.reshape(parent1.weights[i].shape))
            
            current_param += weight_size
            
            # Handle biases
            if current_param + bias_size <= crossover_point:
                child_biases.append(parent1.biases[i].copy())
            elif current_param >= crossover_point:
                child_biases.append(parent2.biases[i].copy())
            else:
                split_point = crossover_point - current_param
                child_bias = np.concatenate([parent1.biases[i][:split_point], 
                                           parent2.biases[i][split_point:]])
                child_biases.append(child_bias)
            
            current_param += bias_size
        
        return network(parent1.X, parent1.Y, parent1.layers[1:], weights=child_weights, biases=child_biases)
    
    @staticmethod
    def crossover_multi_point(parent1, parent2, n_points=2):
        child_weights = []
        child_biases = []
        
        # Calculate total number of parameters
        total_params = sum(w.size + b.size for w, b in zip(parent1.weights, parent1.biases))
        
        # Generate sorted crossover points
        crossover_points = sorted(np.random.choice(range(1, total_params), 
                                                 size=min(n_points, total_params-1), 
                                                 replace=False))
        crossover_points = [0] + crossover_points + [total_params]
        
        # Flatten all parameters
        flat_p1 = np.concatenate([w.flatten() for w in parent1.weights] + 
                                [b.flatten() for b in parent1.biases])
        flat_p2 = np.concatenate([w.flatten() for w in parent2.weights] + 
                                [b.flatten() for b in parent2.biases])
        
        # Create child by alternating between parents
        child_flat = np.zeros_like(flat_p1)
        for i in range(len(crossover_points) - 1):
            start, end = crossover_points[i], crossover_points[i + 1]
            if i % 2 == 0:
                child_flat[start:end] = flat_p1[start:end]
            else:
                child_flat[start:end] = flat_p2[start:end]
        
        # Reconstruct weights and biases
        current_idx = 0
        for i in range(len(parent1.weights)):
            weight_size = parent1.weights[i].size
            bias_size = parent1.biases[i].size
            
            child_weights.append(child_flat[current_idx:current_idx + weight_size]
                               .reshape(parent1.weights[i].shape))
            current_idx += weight_size
            
            child_biases.append(child_flat[current_idx:current_idx + bias_size])
            current_idx += bias_size
        
        return network(parent1.X, parent1.Y, parent1.layers[1:], weights=child_weights, biases=child_biases)
    
    # Each parameter chosen randomly from either parent, prob is probability of choosing a parameter from parent1
    @staticmethod
    def crossover_uniform(parent1, parent2, prob=0.5):
        child_weights = []
        child_biases = []
        
        for i in range(len(parent1.weights)):
            # Uniform crossover for weights
            mask = np.random.random(parent1.weights[i].shape) < prob
            child_weight = np.where(mask, parent1.weights[i], parent2.weights[i]) # If mask == True, use parent1, if False use parent2
            child_weights.append(child_weight)
            
            # Uniform crossover for biases
            mask = np.random.random(parent1.biases[i].shape) < prob
            child_bias = np.where(mask, parent1.biases[i], parent2.biases[i])
            child_biases.append(child_bias)
        
        return network(parent1.X, parent1.Y, parent1.layers[1:], weights=child_weights, biases=child_biases)
    
    @staticmethod
    def crossover(parent1, parent2, method='average', **kwargs):
        if method == 'average':
            return network.crossover_average(parent1, parent2)
        elif method == 'single_point':
            return network.crossover_single_point(parent1, parent2)
        elif method == 'multi_point':
            return network.crossover_multi_point(parent1, parent2, kwargs.get('n_points', 2))
        elif method == 'uniform':
            return network.crossover_uniform(parent1, parent2, kwargs.get('prob', 0.5))
        else:
            raise ValueError(f"Unknown crossover method: {method}")
    
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
    def evolution(X, Y, layers=None, generations=100, pop_size=20, k=3, mutation_rate=0.1, elitism_rate=0.05, crossover_method='average',
                  crossover_kwargs=None, adaptive_mutation=True, early_stopping=None):

        if crossover_kwargs is None:
            crossover_kwargs = {}
            
        # Initialize population
        population = network.initialize_population(pop_size, X, Y, layers)
        
        # Track fitness history for adaptive mutation
        prev_best_fitness = float('inf')
        stagnation_count = 0
        current_mutation_rate = mutation_rate
        
        for gen in range(generations):
            # Sort population by fitness (lower is better)
            population.sort(key=lambda net: net.fitness())
            
            current_best_fitness = population[0].fitness()
            fitness_improvement = prev_best_fitness - current_best_fitness
            
            # Update adaptive mutation rate
            if adaptive_mutation:
                if fitness_improvement <= 1e-6:
                    stagnation_count += 1
                else:
                    stagnation_count = 0
                
                current_mutation_rate = network.adaptive_mutation_rate(
                    mutation_rate, fitness_improvement, stagnation_count
                )
            
            # Determine number of elites (at least 1)
            elitism = max(1, int(elitism_rate * pop_size))
            
            # Copy elites into the new population
            new_population = population[:elitism]
            
            # Fill the rest with offspring
            while len(new_population) < pop_size:
                p1, p2 = network.select_parents(population, k)
                child = network.crossover(p1, p2, method=crossover_method, **crossover_kwargs)
                child.mutate(rate=current_mutation_rate)
                new_population.append(child)
            
            population = new_population
            
            # Track best fitness
            best_net = population[0]
            best_fitness = best_net.fitness()
            
            print(f"Generation {gen+1}, best fitness: {best_fitness}, "
                      f"mutation rate: {current_mutation_rate}, "
                      f"stagnation: {stagnation_count}")
            
            prev_best_fitness = current_best_fitness
            
            if best_fitness < 1e-3:
                break
            if early_stopping is not None and stagnation_count >= early_stopping:
                break
        
        return population[0]
    
    @staticmethod
    def distance():
        pass

net1 = network(X_XOR, Y_XOR, layers=[2, 2, 1])
net1.print_genome()

'''
best_net_avg = network.evolution(X, Y, layers=[2, 2, 1], generations=1000, pop_size=1000, k=10, mutation_rate=0.15, elitism_rate=0.01, 
                                crossover_method='average', adaptive_mutation=True)

best_net_single = network.evolution(X, Y, layers=[2, 2, 1], generations=1000, pop_size=1000, k=10, mutation_rate=0.15, elitism_rate=0.01, 
                                   crossover_method='single_point', adaptive_mutation=True)

best_net_multi = network.evolution(X, Y, layers=[2, 2, 1], generations=1000, pop_size=1000, k=10, mutation_rate=0.15, elitism_rate=0.01, 
                                  crossover_method='multi_point', crossover_kwargs={'n_points': 3}, adaptive_mutation=True)

best_net_uniform = network.evolution(X, Y, layers=[2, 2, 1], generations=1000, pop_size=1000, k=10, mutation_rate=0.15, elitism_rate=0.01, 
                                    crossover_method='uniform', crossover_kwargs={'prob': 0.6}, adaptive_mutation=True)

print(f"\nFinal Results:")
print(f"Average crossover fitness: {best_net_avg.fitness()}")
print(f"Single-point crossover fitness: {best_net_single.fitness()}")
print(f"Multi-point crossover fitness: {best_net_multi.fitness()}")
print(f"Uniform crossover fitness: {best_net_uniform.fitness()}")
'''