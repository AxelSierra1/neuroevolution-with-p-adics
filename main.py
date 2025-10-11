from sage.all import *
import numpy as np

from nn_reals.Network import Network

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
x = np.linspace(-np.pi, np.pi, 10).reshape(-1, 1)  # 100 points between -π and π
# Define output function
y = 0.5 * np.cos(2 * x)




# best_net_avg = Network.evolution(x, y, layers=[2, 2, 1], generations=500, pop_size=500, k=5, mutation_rate=0.15, elitism_rate=0.01, 
#                                crossover_method='average', adaptive_mutation=True)

# best_net_multi = Network.evolution(x, y, layers=[2, 2, 1], generations=500, pop_size=500, k=5, mutation_rate=0.15, elitism_rate=0.01, 
#                                  crossover_method='point', crossover_kwargs={'n_points': 1}, adaptive_mutation=True)

# best_net_uniform = Network.evolution(x, y, layers=[2, 1], generations=500, pop_size=500, k=5, mutation_rate=0.1, elitism_rate=0.1, 
#                                     crossover_method='uniform', crossover_kwargs={'prob': 0.5}, adaptive_mutation=True)

best_net1 = Network.evolution(x, y, layers=[8, 4, 1], generations=1000)

print("labels: ", y)
print("Predictions: ", best_net1.output())