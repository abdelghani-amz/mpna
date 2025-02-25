import matplotlib.pyplot as plt
import re

def plot_convergence(filename, title):
    with open(filename) as f:
        lines = f.readlines()
    
    # Extract bounds from the first line
    bounds_str = lines[0].split(': ')[1]
    bounds = re.findall(r'[-+]?\d*\.?\d+', bounds_str)
    lower = float(bounds[0])
    upper = float(bounds[1])
    
    # Extract eigenvalue approximations
    data = [float(line.strip()) for line in lines[1:]]
    iterations = list(range(1, len(data) + 1))
    
    # Plot settings
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, data, label='Approximated Eigenvalue')
    plt.axhline(lower, color='r', linestyle='--', label='Lower Bound')
    plt.axhline(upper, color='g', linestyle='--', label='Upper Bound')
    plt.xlabel('Iteration')
    plt.ylabel('Eigenvalue Approximation')
    plt.title(f'Convergence of Power Iteration ({title})')
    plt.legend()
    plt.grid(True)
    plt.ylim(lower + lower/10, upper + upper/10)  # Ensure bounds are visible
    plt.show()

# Generate plots for both files
plot_convergence('bcss.txt', 'bcsstk03')
plot_convergence('cfd.txt', 'cfd1')