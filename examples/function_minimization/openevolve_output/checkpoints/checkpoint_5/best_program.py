# EVOLVE-BLOCK-START
"""Function minimization example for OpenEvolve"""
import numpy as np


def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """
    Hybrid simulated annealing with local refinement to escape local minima.

    Args:
        iterations: Number of iterations to run
        bounds: Bounds for the search space (min, max)

    Returns:
        Tuple of (best_x, best_y, best_value)
    """
    # Multiple restarts from different starting points
    num_restarts = 5
    restart_iterations = iterations // num_restarts
    
    best_global_x = np.random.uniform(bounds[0], bounds[1])
    best_global_y = np.random.uniform(bounds[0], bounds[1])
    best_global_value = evaluate_function(best_global_x, best_global_y)
    
    for restart in range(num_restarts):
        # Initialize restart point
        if restart == 0:
            x, y = best_global_x, best_global_y
        else:
            x = np.random.uniform(bounds[0], bounds[1])
            y = np.random.uniform(bounds[0], bounds[1])
        
        current_value = evaluate_function(x, y)
        best_restart_value = current_value
        best_restart_x, best_restart_y = x, y
        
        # Initial temperature for simulated annealing
        temp = 2.0
        temp_decay = 0.95
        
        for i in range(restart_iterations):
            # Adaptive step size based on iteration progress
            step_scale = max(0.1, 2.0 * (1 - i/restart_iterations))
            
            # Generate candidate with local search bias
            if i < restart_iterations * 0.7:
                # Global exploration with simulated annealing
                dx = np.random.normal(0, step_scale)
                dy = np.random.normal(0, step_scale)
            else:
                # Local refinement around best point
                dx = np.random.normal(0, step_scale * 0.3)
                dy = np.random.normal(0, step_scale * 0.3)
            
            new_x = np.clip(x + dx, bounds[0], bounds[1])
            new_y = np.clip(y + dy, bounds[0], bounds[1])
            new_value = evaluate_function(new_x, new_y)
            
            # Accept or reject based on simulated annealing
            if new_value < current_value:
                # Always accept better solutions
                x, y = new_x, new_y
                current_value = new_value
            else:
                # Sometimes accept worse solutions to escape local minima
                delta = new_value - current_value
                prob = np.exp(-delta / temp)
                if np.random.random() < prob:
                    x, y = new_x, new_y
                    current_value = new_value
            
            # Update best for this restart
            if current_value < best_restart_value:
                best_restart_value = current_value
                best_restart_x, best_restart_y = x, y
            
            # Cool down temperature
            temp *= temp_decay
        
        # Update global best
        if best_restart_value < best_global_value:
            best_global_value = best_restart_value
            best_global_x, best_global_y = best_restart_x, best_restart_y
    
    return best_global_x, best_global_y, best_global_value


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
