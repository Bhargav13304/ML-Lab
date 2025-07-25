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


def minimax(depth, node_index, is_maximizing_player, scores, target_depth):
    if depth == target_depth:
        return scores[node_index]

    if is_maximizing_player:
        return max(
            minimax(depth + 1, node_index * 2, False, scores, target_depth),
            minimax(depth + 1, node_index * 2 + 1, False, scores, target_depth)
        )
    else:
        return min(
            minimax(depth + 1, node_index * 2, True, scores, target_depth),
            minimax(depth + 1, node_index * 2 + 1, True, scores, target_depth)
        )


def alphabeta(depth, node_index, is_maximizing_player, scores, target_depth, alpha, beta):
    if depth == target_depth:
        return scores[node_index]

    if is_maximizing_player:
        max_eval = float('-inf')
        for i in range(2):
            eval = alphabeta(depth + 1, node_index * 2 + i, False, scores, target_depth, alpha, beta)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cut-off
        return max_eval
    else:
        min_eval = float('inf')
        for i in range(2):
            eval = alphabeta(depth + 1, node_index * 2 + i, True, scores, target_depth, alpha, beta)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cut-off
        return min_eval


# ---------------- Input Section ----------------
print("Enter the depth of the game tree (e.g., 3 for 8 leaf nodes):")
tree_depth = int(input("Depth: "))
num_leaves = 2 ** tree_depth

print(f"Enter {num_leaves} leaf node scores separated by space:")
scores_input = input("Scores: ")
scores = list(map(int, scores_input.strip().split()))

if len(scores) != num_leaves:
    print(f"Error: Expected {num_leaves} scores, but got {len(scores)}.")
else:
    optimal_value = minimax(0, 0, True, scores, tree_depth)
    print(f"\nOptimal value using Minimax: {optimal_value}")

    optimal_value_ab = alphabeta(0, 0, True, scores, tree_depth, float('-inf'), float('inf'))
    print(f"Optimal value using Alpha-Beta Pruning: {optimal_value_ab}")
