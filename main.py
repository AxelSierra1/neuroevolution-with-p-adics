from sage.all import * # For later use
import numpy as np
import matplotlib.pyplot as plt

from nn_reals.Network import Network
from nn_reals.Population import Population
from nn_reals.Neuroevolution import Neuroevolution
from nn_reals.TreeVisualization import HierarchicalClusteringTree

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
x = np.linspace(-np.pi, np.pi, 10).reshape(-1, 1)  # 10 points between -π and π
# Define output function
y = 0.5 * np.cos(2 * x ** 2) * x



pop = Population(x, y, layers=[2, 1], task='regression', pop_size=25)
evolve = Neuroevolution(pop)
best_net = evolve.evolution(generations=1000)
print("labels: ", y)
print("Predictions: ", best_net.output())
# print("Population size: ", len(pop))
# print("First individual in population: ", pop[0])




# Visualzations =============================================================
# Example usage with auto-generated thresholds:
tree = HierarchicalClusteringTree.from_neuroevolution(
    evolve, 
    generation=999,
    metric='euclidean', 
    auto_thresholds=5
)
tree.build_tree()
tree.verify_clusters()  # Verify that constraints are satisfied
tree.visualize(save_path='clustering_tree.png')

# Or with manually specified decreasing thresholds:
# tree = HierarchicalClusteringTree.from_neuroevolution(
#     evolve, 
#     generation=999, 
#     metric='euclidean', 
#     thresholds=[1.0, 0.2, 0.04, 0.008, 0.0016, 0.00032] # Must be decreasing
# )
# tree.build_tree()
# tree.visualize(save_path='clustering_tree_999_manual.png')
# tree.verify_clusters()
# ===========================================================================


# Disntaces:
    # Genotypic Distance
    # Phenotypic (Behavioral) Distance
    # Loss/fitness distance
# Distance thresholds (d0, d1, d2, ...):
    # Quantile-based
    # Linear decay
    # Exponential decay
# Temporal handling:
    # Snapshot mode: Visualize a single generation as a static tree. Simplest to start with.
    # Animated mode: Show trees evolving over time. This reveals how clusters form, merge, and diverge—much richer but more complex to implement and render.
    # Lineage mode: Track individuals across generations and show which ancestral cluster they belong to. Adds genealogical information back in.
# Visual encoding:
    # Node size: Population size
    # Node color: Average fitness