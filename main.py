from sage.all import * # For later use
import numpy as np
import matplotlib.pyplot as plt

from nn_reals.Network import Network
from nn_reals.Population import Population
from nn_reals.Neuroevolution import Neuroevolution
from nn_reals.SpeciationAnalyzer import SpeciationAnalyzer

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
y = 0.5 * np.cos(2 * x) * x


# best_net_avg = Network.evolution(x, y, layers=[2, 2, 1], generations=500, pop_size=500, k=5, mutation_rate=0.15, elitism_rate=0.01, 
#                                crossover_method='average', adaptive_mutation=True)

# best_net_multi = Network.evolution(x, y, layers=[2, 2, 1], generations=500, pop_size=500, k=5, mutation_rate=0.15, elitism_rate=0.01, 
#                                  crossover_method='point', crossover_kwargs={'n_points': 1}, adaptive_mutation=True)

# best_net_uniform = Network.evolution(x, y, layers=[2, 1], generations=500, pop_size=500, k=5, mutation_rate=0.1, elitism_rate=0.1, 
#                                     crossover_method='uniform', crossover_kwargs={'prob': 0.5}, adaptive_mutation=True)

pop = Population(x, y, layers=[4, 2, 1], task='regression', pop_size=50)

evolve = Neuroevolution(pop)

best_net = evolve.evolution(generations=50)

print("labels: ", y)
print("Predictions: ", best_net.output())

# print("Population size: ", len(pop))
# print("First individual in population: ", pop[0])

# Speciation Analysis
analyzer = SpeciationAnalyzer(evolve.pop_history, distance_metric='euclidean')

# Find appropriate threshold
thresholds = analyzer.suggest_threshold(gen_idx=0)
threshold = thresholds['Moderate (balanced)']

# Generate visualizations
fig1, data1 = analyzer.plot_phylogenetic_tree(gen_idx=0, threshold=threshold)
fig2 = analyzer.plot_heatmap_with_species(gen_idx=0, threshold=threshold)
fig3 = analyzer.plot_speciation_over_time(threshold=threshold)

plt.show()  # Display all plots

print(analyzer.get_species_summary(gen_idx=0, threshold=threshold))