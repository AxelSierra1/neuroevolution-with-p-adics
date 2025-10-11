import numpy as np

# Neural network with arbitrary layer sizes
# When specifying layers, it should include the ouput layer always
# For binary classification it uses sigmoid (outputs values [0, 1])
# For regression 
class Network:
    def __init__(self, X, Y, layers=None, genome=None, task='regression'):
        self.X = X # Features
        self.Y = Y # Labels
        self.task = task # 'regression' or 'classification'
        
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

    def _sigmoid(self, input):
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
    # The predictions from the nn
    def output(self):
        weights, biases = self.decode_genome()
        current_input = self.X
        for i in range(len(weights)): # Computes the layer operations in a loop until it reaches the output layer
            z = np.dot(current_input, weights[i]) + biases[i]

            # Apply activation based on layer type
            if i < len(weights) - 1:
                current_input = self._sigmoid(z) # Hidden layers always use sigmoid
            else: # Output layer: depends on task
                if self.task == 'classification': # For classification: sigmoid for binary
                    current_input = self._sigmoid(z)
                else: # For regression: linear activation (no activation function)
                    current_input = z

        return current_input # Returns a matrix containing the outputs of all possible entries (all our data) (each row corresponds to a data point)
    
    # Function to determine how good the network is
    # Fitness for regression should be mse, for classification should be cross-entropy
    # Generally, higher value is better, but in this implementation, lower is better (using fitness as an error function)
    def fitness(self):
        if self.task == 'regression':
            return np.mean((self.Y - self.output()) ** 2)
        else:
            predictions = self.output()
            predictions = np.clip(predictions, 1e-15, 1 - 1e-15) # Clip to avoid log(0)
            return -np.mean(self.Y * np.log(predictions) + (1 - self.Y) * np.log(1 - predictions))
    
    # Adds perturbations/noise to the weights and biases
    # rate tells us how strong the noise is
    def mutate(self, rate=0.1, prob=0.1):
        noise = np.random.randn(self.genome.shape[0]) * rate
        mask = np.random.rand(self.genome.shape[0]) < prob
        
        self.genome += noise * mask