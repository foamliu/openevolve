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
    
    # Memory of promising regions (store top candidates)
    promising_regions = [(best_global_x, best_global_y, best_global_value)]
    
    for restart in range(num_restarts):
        # Smart restart initialization
        if restart == 0:
            x, y = best_global_x, best_global_y
        else:
            # Mix of random restart, perturbation, and promising region exploration
            rand_val = np.random.random()
            if restart <= 2 or rand_val < 0.5:
                # Random restart for exploration
                x = np.random.uniform(bounds[0], bounds[1])
                y = np.random.uniform(bounds[0], bounds[1])
            elif rand_val < 0.8 and len(promising_regions) > 1:
                # Explore around promising regions
                region_idx = np.random.randint(0, len(promising_regions))
                region_x, region_y, _ = promising_regions[region_idx]
                perturb_scale = 0.3 / (restart + 1)
                x = np.clip(region_x + np.random.normal(0, perturb_scale), bounds[0], bounds[1])
                y = np.clip(region_y + np.random.normal(0, perturb_scale), bounds[0], bounds[1])
            else:
                # Perturbation around best for exploitation
                perturb_scale = 0.5 / (restart + 1)  # Smaller perturbations in later restarts
                x = np.clip(best_global_x + np.random.normal(0, perturb_scale), bounds[0], bounds[1])
                y = np.clip(best_global_y + np.random.normal(0, perturb_scale), bounds[0], bounds[1])
        
        current_value = evaluate_function(x, y)
        best_restart_value = current_value
        best_restart_x, best_restart_y = x, y
        
        # Adaptive temperature schedule
        temp = 2.5 - 0.3 * restart  # Start cooler in later restarts
        temp_decay = 0.92 + 0.06 * (restart / num_restarts)  # Slower decay in later restarts
        accept_count = 0
        reject_count = 0
        
        for i in range(restart_iterations):
            # Dynamic step size with momentum-based adaptation
            progress = i / restart_iterations
            base_scale = max(0.05, 2.5 * (1 - progress ** 0.8))
            
            # Increase step size if we're stuck (not accepting many moves)
            if i > 50 and accept_count < reject_count * 0.1:
                base_scale *= 1.5
            
            # Adaptive step size based on local gradient
            if i > 10:
                # Estimate local gradient magnitude
                eps = 0.01
                grad_x = (evaluate_function(x + eps, y) - evaluate_function(x - eps, y)) / (2 * eps)
                grad_y = (evaluate_function(x, y + eps) - evaluate_function(x, y - eps)) / (2 * eps)
                grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                
                # Smaller steps in steep regions, larger in flat regions
                if grad_magnitude > 1.0:
                    base_scale *= 0.7
                elif grad_magnitude < 0.1:
                    base_scale *= 1.3
            
            step_scale = base_scale
            
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
            
            # Enhanced acceptance with adaptive criteria
            if new_value < current_value:
                # Always accept better solutions
                x, y = new_x, new_y
                current_value = new_value
                accept_count += 1
            else:
                # Adaptive acceptance for worse solutions
                delta = new_value - current_value
                
                # Increase acceptance probability if we're stuck
                stuck_factor = max(1.0, (reject_count + 1) / (accept_count + 1))
                prob = np.exp(-delta / (temp * stuck_factor))
                
                if np.random.random() < prob:
                    x, y = new_x, new_y
                    current_value = new_value
                    accept_count += 1
                else:
                    reject_count += 1
            
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
        
        # Store promising regions for future restarts
        if best_restart_value < -1.0:  # Threshold for "good" solutions
            promising_regions.append((best_restart_x, best_restart_y, best_restart_value))
            # Keep only top 5 promising regions
            promising_regions = sorted(promising_regions, key=lambda x: x[2])[:5]
    
    # Final local refinement around best solution
    final_iterations = max(50, iterations // 20)
    x, y = best_global_x, best_global_y
    best_value = best_global_value
    
    # Coordinate descent for more systematic refinement
    for i in range(final_iterations):
        step_size = 0.01 * (1 - i/final_iterations)
        
        # Alternate between x and y dimensions
        if i % 2 == 0:
            # Refine x coordinate
            for dx in [-step_size, 0, step_size]:
                new_x = np.clip(x + dx, bounds[0], bounds[1])
                new_value = evaluate_function(new_x, y)
                if new_value < best_value:
                    x = new_x
                    best_value = new_value
        else:
            # Refine y coordinate
            for dy in [-step_size, 0, step_size]:
                new_y = np.clip(y + dy, bounds[0], bounds[1])
                new_value = evaluate_function(x, new_y)
                if new_value < best_value:
                    y = new_y
                    best_value = new_value
    
    return x, y, best_value


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
