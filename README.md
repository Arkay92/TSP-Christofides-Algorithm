# TSP-Christofides-Algorithm-With-Integer-Linear-Programming
This Python script provides an implementation of both an approximate and an exact solution for the Travelling Salesman Problem (TSP). The TSP is a combinatorial optimization problem where the goal is to find the shortest possible route that visits each city exactly once and returns to the original city.

For the approximate solution, the script constructs a complete graph from the given set of points, finds a Minimum Spanning Tree (MST) using Kruskal's algorithm, performs a minimum weight perfect matching for vertices with odd degrees, and finally, finds an Eulerian tour to approximate a solution to TSP. This approach is based on heuristic methods to quickly find a near-optimal solution.

Alternatively, for smaller instances (with three or fewer cities), the script switches to an exact solution using Integer Linear Programming (ILP). It formulates the TSP as an ILP problem, where decision variables represent whether an edge is included in the tour, subject to constraints ensuring that each city is visited exactly once and that the tour is closed.

The approximate solution is faster but may not always yield the optimal solution, while the exact solution guarantees optimality but may be computationally expensive for larger instances.

To use the script, specify the method parameter as either 'heuristic' or 'exact' when calling the tsp function if no parameter is given default will be 'heuristic'.

## Requirements
- Python 3.x
- NetworkX library for handling graph operations, especially for the minimum weight perfect matching.

## Installation
First, ensure that you have Python installed on your system. If not, download and install Python from [python.org](https://www.python.org/).

Then, install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage
To use this script, prepare your dataset as a list of tuples where each tuple represents the coordinates of a point (x, y). Then, pass this dataset to the tsp function.

Example:
```from tsp_solver import tsp

# Define your dataset
data = [(1380, 939), (2848, 96), (3510, 1671)]  # Add more points as needed

# Solve TSP (method param exact or heuristic)
length, path = tsp(data, "exact")

print(f"Path: {path}")
print(f"Total length: {length}")
```

## Contributing
Contributions to improve this TSP solver are welcome. Please feel free to fork the repository, make your changes, and submit a pull request.
