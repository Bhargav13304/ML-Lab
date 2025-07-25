import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Generate some random n-dimensional data
np.random.seed(42)
n = 100
data = pd.DataFrame({
 'X': np.random.normal(0, 1, n),
 'Y': np.random.normal(0, 1, n),
 'Z': np.random.normal(0, 1, n),
 'Category': np.random.choice(['A', 'B', 'C'], n)
})
# -------- Scatter Plot (2D) --------
def scatter_plot():
 plt.figure(figsize=(6, 4))
 sns.scatterplot(data=data, x='X', y='Y', hue='Category')
 plt.title("2D Scatter Plot")
 plt.show()
def box_plot():
 plt.figure(figsize=(6, 4))
 sns.boxplot(data=data, x='Category', y='Z')
 plt.title("Box Plot of Z by Category")
 plt.show()
# -------- Heatmap --------
def heatmap():
 correlation = data[['X', 'Y', 'Z']].corr()
 plt.figure(figsize=(5, 4))
 sns.heatmap(correlation, annot=True, cmap='coolwarm')
 plt.title("Heatmap of Correlation Matrix")
 plt.show()
# -------- Contour Plot --------
def contour_plot():
 x = np.linspace(-3, 3, 100)
 y = np.linspace(-3, 3, 100)
 X, Y = np.meshgrid(x, y)
 Z = np.sin(X*2 + Y*2)
 plt.figure(figsize=(6, 5))
 cp = plt.contourf(X, Y, Z, cmap='viridis') 
 plt.colorbar(cp)
 plt.title("Contour Plot of sin(X² + Y²)")
 plt.xlabel("X")
 plt.ylabel("Y")
 plt.show()
# -------- 3D Surface Plot --------
def surface_plot():
 fig = plt.figure(figsize=(8, 6))
 ax = fig.add_subplot(111, projection='3d')
 x = np.linspace(-3, 3, 100)
 y = np.linspace(-3, 3, 100)
 X, Y = np.meshgrid(x, y)
 Z = np.sin(np.sqrt(X*2 + Y*2))
 surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
 fig.colorbar(surf)
 ax.set_title("3D Surface Plot")
 plt.show()
# -------- Call All Plots --------
scatter_plot()
box_plot()
heatmap()
contour_plot()
surface_plot()



def best_first_search(graph, start, goal, heuristic, path=[]):
    open_list = [(0, start)]
    closed_list = set()
    closed_list.add(start)

    while open_list:
        open_list.sort(key=lambda x: heuristic[x[1]], reverse=True)
        cost, node = open_list.pop()
        path.append(node)

        if node == goal:
            return cost, path

        closed_list.add(node)
        for neighbour, neighbour_cost in graph[node]:
            if neighbour not in closed_list:
                closed_list.add(neighbour)
                open_list.append((cost + neighbour_cost, neighbour))

    return None

# Define the graph
graph = {
    'A': [('B', 11), ('C', 14), ('D', 7)],
    'B': [('A', 11), ('E', 15)],
    'C': [('A', 14), ('E', 8), ('D', 18), ('F', 10)],
    'D': [('A', 7), ('F', 25), ('C', 18)],
    'E': [('B', 15), ('C', 8), ('H', 9)],
    'F': [('G', 20), ('C', 10), ('D', 25)],
    'G': [],
    'H': [('E', 9), ('G', 10)]
}

# Define heuristic values
heuristic = {
    'A': 40,
    'B': 32,
    'C': 25,
    'D': 35,
    'E': 19,
    'F': 17,
    'G': 0,
    'H': 10
}

# Start and goal
start = 'A'
goal = 'G'

# Run the algorithm
result = best_first_search(graph, start, goal, heuristic)
if result:
    print(f"Minimum cost path from {start} to {goal} is {result[1]}")
    print(f"Cost: {result[0]}")
else:
    print(f"No path from {start} to {goal}")
