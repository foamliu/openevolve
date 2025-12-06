# EVOLVE-BLOCK-START
"""Function minimization example for OpenEvolve"""
import numpy as np


def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """
    Efficient hybrid search with multiple candidates and adaptive cooling.
    """
    # Initialize 8 diverse candidates with Latin Hypercube sampling
    candidates = []
    for i in range(8):
        # Latin Hypercube sampling for better coverage
        x = bounds[0] + (i + 0.5) * (bounds[1] - bounds[0]) / 8 + np.random.uniform(-0.3, 0.3)
        y = bounds[0] + np.random.uniform(0, 8) * (bounds[1] - bounds[0]) / 8 + np.random.uniform(-0.3, 0.3)
        x = np.clip(x, bounds[0], bounds[1])
        y = np.clip(y, bounds[0], bounds[1])
        candidates.append([x, y, evaluate_function(x, y)])
    
    # Sort by value and preserve elite
    candidates.sort(key=lambda x: x[2])
    best_x, best_y, best_value = candidates[0]
    elite = candidates[0]  # Preserve best solution
    
    # Simulated annealing parameters
    temp = 2.0
    temp_decay = 0.995
    min_temp = 0.01
    
    for i in range(iterations):
        # Select random candidate for exploration
        idx = np.random.randint(0, len(candidates))
        x, y, val = candidates[idx]
        
        # Adaptive step size based on temperature and progress
        step_scale = min(1.0, temp) * (1 - i/iterations * 0.8)
        step_size = step_scale * (bounds[1] - bounds[0]) / 5
        
        # Mix exploration strategies based on temperature and progress
        if temp > 0.5 or np.random.random() < 0.3:
            # Heavy-tailed Cauchy distribution for better exploration
            new_x = x + np.random.standard_cauchy() * step_size * 0.5
            new_y = y + np.random.standard_cauchy() * step_size * 0.5
        else:
            # Gaussian for fine-tuning
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.normal(0, step_size)
            new_x = x + distance * np.cos(angle)
            new_y = y + distance * np.sin(angle)
        new_val = evaluate_function(new_x, new_y)
        
        # Accept or reject based on simulated annealing
        if new_val < val or np.random.random() < np.exp(-(new_val - val) / temp):
            candidates[idx] = [new_x, new_y, new_val]
            
            # Update global best
            if new_val < best_value:
                best_x, best_y, best_value = new_x, new_y, new_val
        
        # Enhanced local search with gradient information
        if temp < 0.2 and i % 15 == 0:
            # Compute approximate gradient
            eps = 0.001
            dx = (evaluate_function(best_x + eps, best_y) - evaluate_function(best_x - eps, best_y)) / (2 * eps)
            dy = (evaluate_function(best_x, best_y + eps) - evaluate_function(best_x, best_y - eps)) / (2 * eps)
            
            # Multiple local search strategies
            for strategy in range(5):
                if strategy < 2:
                    # Gradient-based steps
                    step_size = 0.01 * (1 - strategy * 0.5)
                    lx = best_x - step_size * dx + np.random.normal(0, 0.01)
                    ly = best_y - step_size * dy + np.random.normal(0, 0.01)
                else:
                    # Random local exploration with decreasing radius
                    radius = 0.05 * (0.5 ** (strategy - 2))
                    angle = np.random.uniform(0, 2 * np.pi)
                    distance = np.random.normal(0, radius)
                    lx = best_x + distance * np.cos(angle)
                    ly = best_y + distance * np.sin(angle)
                
                lx = np.clip(lx, bounds[0], bounds[1])
                ly = np.clip(ly, bounds[0], bounds[1])
                lval = evaluate_function(lx, ly)
                if lval < best_value:
                    best_x, best_y, best_value = lx, ly, lval
                    # Update elite if we found better solution
                    elite = [best_x, best_y, best_value]
        
        # Cool down temperature
        temp = max(min_temp, temp * temp_decay)
        
        # Intelligent restart when stuck
        if i % 50 == 0 and i > 0:
            # Check if we're making progress
            current_best = min(candidates, key=lambda x: x[2])
            if abs(current_best[2] - best_value) < 1e-6:
                # Replace worst candidates with explorative jumps
                candidates.sort(key=lambda x: x[2])
                for j in range(3):  # Replace 3 worst candidates
                    # Jump around elite or random location
                    if np.random.random() < 0.7:
                        # Jump around elite
                        jump_x = elite[0] + np.random.normal(0, 1.0)
                        jump_y = elite[1] + np.random.normal(0, 1.0)
                    else:
                        # Completely random restart
                        jump_x = np.random.uniform(bounds[0], bounds[1])
                        jump_y = np.random.uniform(bounds[0], bounds[1])
                    
                    jump_x = np.clip(jump_x, bounds[0], bounds[1])
                    jump_y = np.clip(jump_y, bounds[0], bounds[1])
                    candidates[-(j+1)] = [jump_x, jump_y, evaluate_function(jump_x, jump_y)]
    
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
