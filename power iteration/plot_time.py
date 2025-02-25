import matplotlib.pyplot as plt

def plot_time(filename, title):
    processes = []
    times = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            p, t = line.strip().split()
            processes.append(int(p))
            times.append(float(t))
    
    plt.figure(figsize=(8, 5))
    plt.plot(processes, times, marker='o', linestyle='-', color='royalblue', linewidth=2, markersize=8)
    plt.xticks(processes)
    plt.xlabel('Number of Processes')
    plt.ylabel('Execution Time (seconds)')
    plt.title(f'Execution Time vs Processes ({title})')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Plot for bcssTime.txt
plot_time('bcssTime.txt', 'bcsstk03')

# Plot for cfdTime.txt
plot_time('cfdTime.txt', 'cfd1')