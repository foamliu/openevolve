# EVOLVE-BLOCK-START
"""Function minimization example for OpenEvolve"""
import numpy as np


def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """
    Adaptive differential evolution with local search hybrid.
    
    Args:
        iterations: Number of iterations to run
        bounds: Bounds for the search space (min, max)
        
    Returns:
        Tuple of (best_x, best_y, best_value)
    """
    # Population size
    pop_size = 20
    
    # Initialize population
    population = np.random.uniform(bounds[0], bounds[1], (pop_size, 2))
    values = np.array([evaluate_function(x, y) for x, y in population])
    
    # Track best solution
    best_idx = np.argmin(values)
    best_x, best_y = population[best_idx]
    best_value = values[best_idx]
    
    # Adaptive parameters
    mutation_factor = 0.8
    crossover_prob = 0.7
    
    for i in range(iterations):
        # Adaptive parameters based on progress
        progress = i / iterations
        mutation_factor = 0.5 + 0.3 * (1 - progress)
        crossover_prob = 0.5 + 0.2 * (1 - progress)
        
        for j in range(pop_size):
            # Select three random individuals (different from j)
            candidates = list(range(pop_size))
            candidates.remove(j)
            a, b, c = np.random.choice(candidates, 3, replace=False)
            
            # Differential mutation
            mutant = population[a] + mutation_factor * (population[b] - population[c])
            
            # Crossover
            trial = population[j].copy()
            crossover_mask = np.random.random(2) < crossover_prob
            trial[crossover_mask] = mutant[crossover_mask]
            
            # Ensure bounds
            trial = np.clip(trial, bounds[0], bounds[1])
            
            # Evaluate trial
            trial_value = evaluate_function(trial[0], trial[1])
            
            # Selection
            if trial_value < values[j]:
                population[j] = trial
                values[j] = trial_value
                
                # Update global best
                if trial_value < best_value:
                    best_x, best_y = trial[0], trial[1]
                    best_value = trial_value
        
        # Periodic local search around best solution
        if i % 50 == 0 and i > 0:
            local_step = 0.1 * (1 - progress)
            for _ in range(10):
                local_x = best_x + np.random.normal(0, local_step)
                local_y = best_y + np.random.normal(0, local_step)
                local_x = np.clip(local_x, bounds[0], bounds[1])
                local_y = np.clip(local_y, bounds[0], bounds[1])
                local_value = evaluate_function(local_x, local_y)
                
                if local_value < best_value:
                    best_x, best_y = local_x, local_y
                    best_value = local_value
    
    return best_x, best_y, best_value


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def evaluate_function(x, y):
    """The complex function we're trying to minimize"""
    return np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20


def run_search():
    x, y, value = search_algorithm()
    return x, y, value


if __name__ == "__main__":
    x, y, value = run_search()
    print(f"Found minimum at ({x}, {y}) with value {value}")
