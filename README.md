# TSP-Christofides-Algorithm-With-Integer-Linear-Programming
This Python script is an implementation of an approximate solution for the Travelling Salesman Problem (TSP) using graph theory and approximation algorithms. It constructs a complete graph from a given set of points, finds a Minimum Spanning Tree (MST) using Kruskal's algorithm, performs a minimum weight perfect matching for vertices with odd degrees, and finally, finds an Eulerian tour to approximate a solution to TSP.

## Requirements
- Python 3.x
- NetworkX library for handling graph operations, especially for the minimum weight perfect matching.

## Installation
First, ensure that you have Python installed on your system. If not, download and install Python from [python.org](https://www.python.org/).

Then, install the dependencies:

```bash
pip install -r requirements.txt

## Usage
To use this script, prepare your dataset as a list of tuples where each tuple represents the coordinates of a point (x, y). Then, pass this dataset to the tsp function.

Example:
```from tsp_solver import tsp

# Define your dataset
data = [(1380, 939), (2848, 96), (3510, 1671)]  # Add more points as needed

# Solve TSP
length, path = tsp(data)

print(f"Path: {path}")
print(f"Total length: {length}")
```

## Contributing
Contributions to improve this TSP solver are welcome. Please feel free to fork the repository, make your changes, and submit a pull request.
