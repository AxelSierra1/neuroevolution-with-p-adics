import numpy as np
import matplotlib.pyplot as plt
import cProfile
from nn_reals.Network import Network
from nn_reals.Population import Population
from nn_reals.Neuroevolution import Neuroevolution

np.random.seed(45)

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

metrics_small = ['euclidean', 'manhattan', 'qbadic']
multipliers_small = [10]
bases_small = [2]

metrics_large=['euclidean', 'manhattan', 'qbadic']
multipliers_large=[0.1, 0.5, 1, 2, 3, 4, 5, 10, 20, 50]
bases_large=[2, 3, 4, 5, 6, 7, 10, 12]

pop = Population(x_featured, y, layers=[20, 18, 1], task='regression', pop_size=500) # [10, 3, 1]
evolve = Neuroevolution(pop)
best_net = evolve.evolution(generations=3000, verbose=True, crossover_method='point', crossover_kwargs={'n_points': 2},
                            early_stopping=200, mutation_rate=0.03, track_metrics=True, track_matrices=False, qbadic_norm='l1',
                            mutation_prob=0.03, k=3, metrics=metrics_large, multipliers=multipliers_large, qbadic_bases=bases_large,
                            fdc_correlation_type='spearman')

# --- RESULTS ANALYSIS ---
if evolve.ev_metrics:
    # Define logger to write to both console and file
    results_file = "results_summary.txt"
    with open(results_file, "w") as f:
        def log(message):
            print(message)
            f.write(message + "\n")

        log("\n" + "="*50)
        log("EVOLUTION RESULTS ANALYSIS")
        log("="*50)
        
        # 1. Diversity & FDC Summary
        summary = evolve.ev_metrics.get_results_summary()
        
        header = f"\n{'Metric Configuration':<30} | {'Start Div':<10} | {'End Div':<10} | {'Mid 50% Div':<12} | {'Avg FDC':<10} | {'Avg Samples':<11}"
        log(header)
        log("-" * len(header))
        
        for name, stats in summary.items():
            sample_size = stats.get('avg_fdc_sample_size', 0)
            log(f"{name:<30} | {stats['start_diversity']:^10.4f} | {stats['end_diversity']:^10.4f} | "
                f"{stats['mid_50_diversity']:^12.4f} | {stats['avg_fdc']:^10.4f} | {sample_size:^11.1f}")

        # 2. Orthogonality (Metric-Metric Correlation)
        log("\n" + "-"*50)
        log("ORTHOGONALITY SCORE (Metric Independence)")
        log("-"*50)
        log("Correlation between distance matrices (lower = more independent/orthogonal)\n")
        
        # Calculate orthogonality with Pearson (linear relationship) and Spearman (rank relationship)
        ortho_results = evolve.ev_metrics.calculate_orthogonality(
            evolve.population,
            #metric_pairs=[('euclidean', 'manhattan'), ('euclidean', 'qbadic_b2_m10'), ('manhattan', 'qbadic_b2_m10')],
            correlation_type='spearman',
            n_samples=100
        )
        
        for pair, score in ortho_results.items():
            log(f"{pair:<40} : {score:.4f}")
        log("="*50 + "\n")
        
    print(f"Results summary saved to {results_file}")

# Save the best network
best_net.save('best_function_approximator.npz')
# Load the network
# loaded_model = Network.load('best_function_approximator.npz')


# if __name__ == "__main__":
#     cProfile.run('evolve.evolution(generations=5, verbose=True, crossover_method="point", crossover_kwargs={"n_points": 2}, early_stopping=200, mutation_rate=0.05,  mutation_prob=0.05, k=2)')
    
# print("labels: ", y)
# print("Predictions: ", best_net.output())
# print("Population size: ", len(pop))
# print("First individual in population: ", pop[0])

# High correlation means: "Networks that are genetically far apart tend to have different fitness values."
# Average correlations (fitness diff vs distance) refer to the fitness diff between 2 genomes vs their distance according to some norm
# Disntaces:
    # Genotypic Distance
    # Phenotypic (Behavioral) Distance
    # Loss/fitness distance
# Temporal handling:
    # Snapshot mode: Visualize a single generation as a static tree. Simplest to start with.
    # Animated mode: Show trees evolving over time. This reveals how clusters form, merge, and diverge—much richer but more complex to implement and render.
    # Lineage mode: Track individuals across generations and show which ancestral cluster they belong to. Adds genealogical information back in.


# Generate predictions
predictions = best_net.predict(x_featured)
# predictions = loaded_model.predict(x_featured)

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

plt.savefig('function_approximation_plot.png', dpi=300) # dpi for higher resolution
print("Plot saved to function_approximation_plot.png")

plt.show()