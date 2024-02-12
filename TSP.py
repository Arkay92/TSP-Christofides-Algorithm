import pulp
import math
import heapq
import networkx as nx

# Function to solve the Traveling Salesman Problem (TSP)
def tsp(data, method='heuristic'):
    # If the number of data points is small, switch to exact method
    if len(data) <= 3:
        method = 'exact'
    
    # Choose the appropriate method based on input
    if method == 'heuristic':
        return tsp_heuristic(data)
    elif method == 'exact':
        return tsp_exact(data)
    else:
        raise ValueError("Method must be either 'heuristic' or 'exact'")

# Heuristic method for solving TSP
def tsp_heuristic(data):
    # Build graph from given data points
    G = build_graph(data)
    
    # Find Minimum Spanning Tree (MST) using Kruskal's algorithm
    MSTree = minimum_spanning_tree(G)
    
    # Find vertices with odd degrees in MST
    odd_vertexes = find_odd_vertexes(MSTree, G)
    
    # Perform minimum weight perfect matching for odd degree vertices
    augmented_MST = minimum_weight_matching(G, odd_vertexes)
    
    # Find an Eulerian tour in the augmented MST
    eulerian_tour = find_eulerian_tour(augmented_MST, G)
    
    # Create a Hamiltonian circuit from the Eulerian tour
    return create_hamiltonian_circuit(eulerian_tour, G)

# Exact method for solving TSP using Integer Linear Programming (ILP)
def tsp_exact(data):
    n = len(data)
    problem = pulp.LpProblem("TSP", pulp.LpMinimize)

    # Decision variables: x[(i, j)] = 1 if edge (i, j) is selected, 0 otherwise
    x = pulp.LpVariable.dicts("x", [(i, j) for i in range(n) for j in range(n) if i != j], 0, 1, pulp.LpBinary)

    # Objective function: minimize total distance
    problem += pulp.lpSum([x[(i, j)] * get_length(data[i][0], data[i][1], data[j][0], data[j][1]) for i in range(n) for j in range(n) if i != j])

    # Constraints: each city is visited exactly once
    for i in range(n):
        problem += pulp.lpSum([x[(i, j)] for j in range(n) if i != j]) == 1
        problem += pulp.lpSum([x[(j, i)] for j in range(n) if i != j]) == 1

    # Subtour elimination constraints using Miller-Tucker-Zemlin formulation
    u = pulp.LpVariable.dicts("u", range(n), 0, n-1, pulp.LpInteger)
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                problem += u[i] - u[j] + n * x[(i, j)] <= n - 1

    # Solve the ILP problem
    problem.solve()

    # Check if optimal solution is found
    if pulp.LpStatus[problem.status] != "Optimal":
        raise Exception(f"No optimal solution found. Status: {pulp.LpStatus[problem.status]}")

    # Extract tour from the solution
    tour = extract_tour(x, n)
    
    # Calculate total distance of the tour
    total_distance = calculate_total_distance(tour, data)

    return total_distance, tour

# Function to calculate the Euclidean distance between two points
def get_length(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

# Function to build a complete graph from given data points
def build_graph(data):
    graph = {i: {} for i in range(len(data))}
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            graph[i][j] = graph[j][i] = get_length(data[i][0], data[i][1], data[j][0], data[j][1])
    return graph

# Class for implementing Union-Find data structure
class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, i):
        if i not in self.parent:
            self.parent[i] = i
        while i != self.parent[i]:
            i = self.parent[i] = self.parent[self.parent[i]]
        return i

    def union(self, i, j):
        self.parent[self.find(i)] = self.find(j)

# Function to find Minimum Spanning Tree (MST) using Kruskal's algorithm
def minimum_spanning_tree(G):
    edges = [(cost, u, v) for u, neighbors in G.items() for v, cost in neighbors.items() if u < v]
    heapq.heapify(edges)
    tree, subtrees = [], UnionFind()
    while edges and len(tree) < len(G) - 1:
        cost, u, v = heapq.heappop(edges)
        if subtrees.find(u) != subtrees.find(v):
            tree.append((u, v))
            subtrees.union(u, v)
    return tree

# Function to find vertices with odd degrees in MST
def find_odd_vertexes(MST, G):
    degree = {v: 0 for v in G}
    for u, v in MST:
        degree[u] += 1
        degree[v] += 1
    return [v for v, d in degree.items() if d % 2 == 1]

# Function to perform minimum weight perfect matching for odd degree vertices
def minimum_weight_matching(G, odd_vert):
    subgraph = nx.Graph()
    for v in odd_vert:
        for u in odd_vert:
            if u != v:
                subgraph.add_edge(v, u, weight=-G[v][u])  # Negate weight for max weight matching
    matching = nx.max_weight_matching(subgraph, maxcardinality=True)
    return [(u, v) for u, v in matching]

# Function to find an Eulerian tour in the augmented MST
def find_eulerian_tour(MST, G):
    neighbors = {v: [] for v in G}
    for u, v in MST:
        neighbors[u].append(v)
        neighbors[v].append(u)

    start_vertex = max(neighbors, key=lambda v: len(neighbors[v]))
    stack, path = [start_vertex], []
    while stack:
        u = stack[-1]
        if neighbors[u]:
            v = neighbors[u].pop()
            neighbors[v].remove(u)
            stack.append(v)
        else:
            path.append(stack.pop())
    return path

# Function to create a Hamiltonian circuit from an Eulerian tour
def create_hamiltonian_circuit(eulerian_tour, G):
    path, visited = [], set()
    for v in eulerian_tour:
        if v not in visited:
            path.append(v)
            visited.add(v)
    path.append(path[0])  # Complete the circuit
    return path

# Function to extract tour from the ILP solution
def extract_tour(x, n):
    tour = [0]  # Start from the first node
    current = 0
    for _ in range(1, n):
        for j in range(n):
            if x[(current, j)].varValue == 1:
                tour.append(j)
                current = j
                break
    return tour

# Function to calculate the total distance of the tour
def calculate_total_distance(tour, data):
    total_distance = sum(get_length(data[tour[i-1]][0], data[tour[i-1]][1], data[tour[i]][0], data[tour[i]][1]) for i in range(1, len(tour)))
    total_distance += get_length(data[tour[-1]][0], data[tour[-1]][1], data[tour[0]][0], data[tour[0]][1])
    return total_distance
