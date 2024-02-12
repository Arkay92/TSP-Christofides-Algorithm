import pulp
import math
import heapq
import networkx as nx

def tsp(data, method='heuristic'):
    if len(data) <= 3:
        method = 'exact'
    
    if method == 'heuristic':
        return tsp_heuristic(data)
    elif method == 'exact':
        return tsp_exact(data)
    else:
        raise ValueError("Method must be either 'heuristic' or 'exact'")

def tsp_heuristic(data):
    G = build_graph(data)
    MSTree = minimum_spanning_tree(G)
    odd_vertexes = find_odd_vertexes(MSTree, G)
    augmented_MST = minimum_weight_matching(G, odd_vertexes)
    eulerian_tour = find_eulerian_tour(augmented_MST, G)
    return create_hamiltonian_circuit(eulerian_tour, G)

def tsp_exact(data):
    n = len(data)
    problem = pulp.LpProblem("TSP", pulp.LpMinimize)

    x = pulp.LpVariable.dicts("x", [(i, j) for i in range(n) for j in range(n) if i != j], 0, 1, pulp.LpBinary)

    problem += pulp.lpSum([x[(i, j)] * get_length(data[i][0], data[i][1], data[j][0], data[j][1]) for i in range(n) for j in range(n) if i != j])

    for i in range(n):
        problem += pulp.lpSum([x[(i, j)] for j in range(n) if i != j]) == 1
        problem += pulp.lpSum([x[(j, i)] for j in range(n) if i != j]) == 1

    u = pulp.LpVariable.dicts("u", range(n), 0, n-1, pulp.LpInteger)
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                problem += u[i] - u[j] + n*x[(i, j)] <= n-1

    problem.solve()

    if pulp.LpStatus[problem.status] != "Optimal":
        raise Exception(f"No optimal solution found. Status: {pulp.LpStatus[problem.status]}")

    tour = extract_tour(x, n)
    total_distance = calculate_total_distance(tour, data)

    return total_distance, tour

def get_length(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def build_graph(data):
    graph = {i: {} for i in range(len(data))}
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            graph[i][j] = graph[j][i] = get_length(data[i][0], data[i][1], data[j][0], data[j][1])
    return graph

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

def find_odd_vertexes(MST, G):
    degree = {v: 0 for v in G}
    for u, v in MST:
        degree[u] += 1
        degree[v] += 1
    return [v for v, d in degree.items() if d % 2 == 1]

def minimum_weight_matching(G, odd_vert):
    subgraph = nx.Graph()
    for v in odd_vert:
        for u in odd_vert:
            if u != v:
                subgraph.add_edge(v, u, weight=-G[v][u])  # Negate weight for max weight matching
    matching = nx.max_weight_matching(subgraph, maxcardinality=True)
    return [(u, v) for u, v in matching]

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

def create_hamiltonian_circuit(eulerian_tour, G):
    path, visited = [], set()
    for v in eulerian_tour:
        if v not in visited:
            path.append(v)
            visited.add(v)
    path.append(path[0])  # Complete the circuit
    return path

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

def calculate_total_distance(tour, data):
    total_distance = sum(get_length(data[tour[i-1]][0], data[tour[i-1]][1], data[tour[i]][0], data[tour[i]][1]) for i in range(1, len(tour)))
    total_distance += get_length(data[tour[-1]][0], data[tour[-1]][1], data[tour[0]][0], data[tour[0]][1])
    return total_distance
