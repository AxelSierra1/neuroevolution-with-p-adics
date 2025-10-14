import numpy as np

from nn_reals.Network import Network

class Neuroevolution:
    def __init__(self, population):
        self.population = population
        self.pop_history = [self.population]

    # Tournament selection
    # We choose 2 parents from a cluster of k candidates chosen at random (we choose the 2 with best fitness in this cluster)
    def select_parents(self, k):
        # First parent
        candidates1 = np.random.choice(self.population.pop, k, replace=False)
        parent1 = min(candidates1, key=lambda net: net.fitness())

        # Second parent, ensure its different than parent1
        while True:
            candidates2 = np.random.choice(self.population.pop, k, replace=False)
            parent2 = min(candidates2, key=lambda net: net.fitness())
            if parent2 is not parent1:
                break

        return parent1, parent2

    # Simply average all weight matrices and bias vectors of both parents
    def crossover_average(self, parent1, parent2):
        child_genome = (parent1.genome + parent2.genome) / 2
        
        return Network(parent1.X, parent1.Y, layers=parent1.layers[1:], genome=child_genome, task=parent1.task)
    
    # Chooses randomly a point in the list of all parameters of the child, parameters before that point come from parent1, after that point come from parent2
    # Can choose the amount of points to split the genes from both parents
    def crossover_npoints(self, parent1, parent2, n_points=1):
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
        
        return Network(parent1.X, parent1.Y, layers=parent1.layers[1:], genome=child_genome, task=parent1.task)
    
    # Each parameter chosen randomly from either parent, prob is probability of choosing a parameter from parent1
    def crossover_uniform(self, parent1, parent2, prob=0.5):
        # Create uniform crossover mask
        mask = np.random.random(parent1.genome.shape) < prob
        
        # If mask == True, use parent1; if False, use parent2
        child_genome = np.where(mask, parent1.genome, parent2.genome)
        
        return Network(parent1.X, parent1.Y, layers=parent1.layers[1:], genome=child_genome, task=parent1.task)

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
    
    def evolution(self, generations=100, k=3, mutation_rate=0.15, mutation_prob=0.15, elitism_rate=0.05, crossover_method='average',
                  crossover_kwargs=None, adaptive_mutation=True, early_stopping=None, task='regression'):
        
        if crossover_kwargs is None:
            crossover_kwargs = {}
        
        prev_best_fitness = float('inf')
        stagnation_count = 0
        current_mutation_rate = mutation_rate
        
        for gen in range(generations):
            # Sort population by fitness
            self.population.pop.sort(key=lambda net: net.fitness())

            # Calculate diversity each epoch
            diversity_stats = self.population.population_diversity(n_samples=100)
            print(f"Mean diversity: {diversity_stats['mean_distance']:.4f}")
            print(f"Std deviation: {diversity_stats['std_distance']:.4f}")
            
            current_best_fitness = self.population.pop[0].fitness()
            fitness_improvement = prev_best_fitness - current_best_fitness
            
            if adaptive_mutation:
                if fitness_improvement <= 1e-6:
                    stagnation_count += 1
                else:
                    stagnation_count = 0
                
                current_mutation_rate = Neuroevolution.adaptive_mutation_rate(
                    mutation_rate, fitness_improvement, stagnation_count
                )
            
            # Elitism: preserve best networks
            elitism = max(1, int(elitism_rate * self.population.pop_size))
            new_population = self.population.pop[:elitism]
            
            # Generate new offspring
            while len(new_population) < self.population.pop_size:
                p1, p2 = self.select_parents(k)
                
                if crossover_method == 'average':
                    child = self.crossover_average(p1, p2)
                elif crossover_method == 'point':
                    n_points = crossover_kwargs.get('n_points', 1)
                    child = self.crossover_npoints(p1, p2, n_points=n_points)
                elif crossover_method == 'uniform':
                    prob = crossover_kwargs.get('prob', 0.5)
                    child = self.crossover_uniform(p1, p2, prob=prob)
                else:
                    raise ValueError(f"Unknown crossover method: {crossover_method}")
                
                child.mutate(rate=current_mutation_rate, prob=mutation_prob)
                new_population.append(child)
            
            # Update population
            self.population.pop = new_population
            self.pop_history.append(self.population.pop[:])
            best_net = self.population.pop[0]
            best_fitness = best_net.fitness()
            
            print(f"Generation {gen+1}, best fitness: {best_fitness:.6f}, "
                  f"mutation rate: {current_mutation_rate:.4f}, "
                  f"stagnation: {stagnation_count}")
            
            prev_best_fitness = current_best_fitness
            
            # Convergence check
            if best_fitness < 1e-4:
                print("Converged!")
                break
            if early_stopping is not None and stagnation_count >= early_stopping:
                print(f"Early stopping at generation {gen+1}")
                break
        return self.population.pop[0]