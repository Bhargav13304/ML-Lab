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


import heapq

class Node:
    def __init__(self, name, heuristic, parent=None):
        self.name = name
        self.heuristic = heuristic
        self.parent = parent

    def __lt__(self, other):
        return self.heuristic < other.heuristic

def best_first_search(graph, start, goal, heuristic_values):
    open_list = []
    closed_list = set()
    heapq.heappush(open_list, Node(start, heuristic_values[start]))

    while open_list:
        current_node = heapq.heappop(open_list)

        if current_node.name == goal:
            path = []
            while current_node:
                path.append(current_node.name)
                current_node = current_node.parent
            return path[::-1]

        if current_node.name in closed_list:
            continue

        closed_list.add(current_node.name)

        for neighbor in graph.get(current_node.name, []):
            if neighbor not in closed_list:
                heapq.heappush(open_list, Node(neighbor, heuristic_values[neighbor], current_node))

    return None

# Take custom input for the graph and heuristic values
def get_input():
    graph = {}
    heuristic_values = {}

    print("Enter the graph structure:")
    n = int(input("Enter number of nodes: "))

    for _ in range(n):
        node = input("Enter node name: ")
        neighbors = input(f"Enter neighbors for {node} (comma separated): ").split(",")
        graph[node] = [neighbor.strip() for neighbor in neighbors if neighbor.strip()]

    print("\nEnter heuristic values:")
    for _ in range(n):
        node = input("Enter node name for heuristic: ")
        heuristic = int(input(f"Enter heuristic value for {node}: "))
        heuristic_values[node] = heuristic

    start = input("\nEnter the start node: ")
    goal = input("Enter the goal node: ")

    return graph, heuristic_values, start, goal

# Main execution
if __name__ == "__main__":
    graph, heuristic_values, start_node, goal_node = get_input()
    path = best_first_search(graph, start_node, goal_node, heuristic_values)
    if path:
        print(f"\nPath from {start_node} to {goal_node}: {path}")
    else:
        print(f"\nNo path found from {start_node} to {goal_node}.")
