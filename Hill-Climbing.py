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
 Z = np.sin(X**2 + Y**2)
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
 Z = np.sin(np.sqrt(X**2 + Y**2))
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


import random

# Objective function (maximize this)
def objective_function(x):
    return -x**2 + 5

# Hill Climbing algorithm
def hill_climbing(start_x, step_size, max_iterations):
    current_x = start_x
    current_score = objective_function(current_x)

    for i in range(max_iterations):
        # Try a new solution nearby
        new_x = current_x + random.uniform(-step_size, step_size)
        new_score = objective_function(new_x)

        print(f"Iteration {i+1}: x = {current_x:.4f}, f(x) = {current_score:.4f}")

        # If the new solution is better, move to it
        if new_score > current_score:
            current_x = new_x
            current_score = new_score
        else:
            # No improvement; stop if you want a simple version
            pass

    print("\nFinal Solution:")
    print(f"x = {current_x:.4f}, f(x) = {current_score:.4f}")
    return current_x, current_score

# Run the algorithm
best_x, best_score = hill_climbing(start_x=0.1, step_size=0.05, max_iterations=5)
