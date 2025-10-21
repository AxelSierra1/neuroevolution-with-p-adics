# from sage.all import * # For later use
import numpy as np
import matplotlib.pyplot as plt

from nn_reals.Network import Network
from nn_reals.Population import Population
from nn_reals.Neuroevolution import Neuroevolution

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
x = np.linspace(-np.pi, np.pi, 500).reshape(-1, 1)  # 500 points between -π and π
# Define output function
y = 0.5 * np.cos(2 * x ** 2) * x


pop = Population(x, y, layers=[4, 1, 1], task='regression', pop_size=500)
evolve = Neuroevolution(pop)
best_net = evolve.evolution(generations=1000, verbose=True)
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