# from sage.all import * # For later use
import numpy as np
import matplotlib.pyplot as plt

from nn_reals.Network import Network
from nn_reals.Population import Population
from nn_reals.Neuroevolution import Neuroevolution

import cProfile

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
x = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1)  # 500 points between -π and π
# Define output function
y = 0.5 * np.cos(2 * x ** 2) * x

x_input = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1)
x_featured = np.hstack([
    x_input,          # The original linear term
    x_input**2,       # The quadratic term
    np.sin(x_input),  # A basic periodic component
    np.cos(x_input)   # Another basic periodic component
])

pop = Population(x_featured, y, layers=[20, 18, 1], task='regression', pop_size=1000) # [10, 3, 1]
evolve = Neuroevolution(pop)
best_net = evolve.evolution(generations=3000, verbose=True, crossover_method='point', crossover_kwargs={'n_points': 2},
                            early_stopping=200, mutation_rate=0.03,  mutation_prob=0.03, k=3, metric_interval=5)

# if __name__ == "__main__":
#     cProfile.run('evolve.evolution(generations=5, verbose=True, crossover_method="point", crossover_kwargs={"n_points": 2}, early_stopping=200, mutation_rate=0.05,  mutation_prob=0.05, k=2)')
    
# print("labels: ", y)
# print("Predictions: ", best_net.output())
# print("Population size: ", len(pop))
# print("First individual in population: ", pop[0])


# Pearson correlation across many pairs.
# High correlation means: "Networks that are genetically far apart tend to have different fitness values."
# Average correlations (fitness diff vs distance) refer to the fitness diff between 2 genomes vs their distance according to some norm

# Base vs multiplier Diversity heatmap

# Disntaces:
    # Genotypic Distance
    # Phenotypic (Behavioral) Distance
    # Loss/fitness distance
# Temporal handling:
    # Snapshot mode: Visualize a single generation as a static tree. Simplest to start with.
    # Animated mode: Show trees evolving over time. This reveals how clusters form, merge, and diverge—much richer but more complex to implement and render.
    # Lineage mode: Track individuals across generations and show which ancestral cluster they belong to. Adds genealogical information back in.

    # Generate predictions from the best network

predictions = best_net.predict(x_featured)

# Plot the results
plt.figure(figsize=(10, 6))

# Plot the original function using the original x values
plt.plot(x_input, y, label='Original Function', color='blue', linewidth=2)

# Plot the neural network's approximation
plt.plot(x_input, predictions, label='NN Approximation', color='red', linestyle='--', linewidth=2)

# Optional: Plot the training data points to see the fit
plt.scatter(x_input, y, label='Training Data', color='black', s=10)

# Add titles and labels for clarity
plt.title('Function Approximation using Neuroevolution')
plt.xlabel('Input (x)')
plt.ylabel('Output (y)')
plt.legend()
plt.grid(True)
plt.show()