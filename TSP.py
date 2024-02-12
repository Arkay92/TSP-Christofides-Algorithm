import heapq
import networkx as nx

def tsp(data):
    # Return early for edge cases: no data or single point
    if not data:
        return 0, []
    elif len(data) == 1:
        return 0, [0]

    G = build_graph(data)
    MSTree = minimum_spanning_tree(G)
    odd_vertexes = find_odd_vertexes(MSTree, G)
    augmented_MST = minimum_weight_matching(G, odd_vertexes)
    eulerian_tour = find_eulerian_tour(augmented_MST, G)
    return create_hamiltonian_circuit(eulerian_tour, G)

def get_length(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

def build_graph(data):
    """Build a complete graph from the data points."""
    graph = {i: {} for i in range(len(data))}
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            distance = get_length(data[i][0], data[i][1], data[j][0], data[j][1])
            graph[i][j] = graph[j][i] = distance
    return graph

class UnionFind:
    """Efficiently manage disjoint sets for Kruskal's algorithm."""
    def __init__(self):
        self.parent = {}

    def find(self, i):
        if i not in self.parent:
            self.parent[i] = i
        while i != self.parent[i]:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        root_i, root_j = self.find(i), self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j

def minimum_spanning_tree(G):
    """Construct a minimum spanning tree using Kruskal's algorithm."""
    edges = [(cost, u, v) for u, neighbors in G.items() for v, cost in neighbors.items() if u < v]
    heapq.heapify(edges)
    tree, subtrees = [], UnionFind()
    while edges and len(tree) < len(G) - 1:
        cost, u, v = heapq.heappop(edges)
        if subtrees.find(u) != subtrees.find(v):
            tree.append((u, v, cost))
            subtrees.union(u, v)
    return tree

def find_odd_vertexes(MST, G):
    """Find vertices with odd degrees in the MST."""
    degree = {v: 0 for v in G}
    for u, v, _ in MST:
        degree[u] += 1
        degree[v] += 1
    return [v for v, d in degree.items() if d % 2 == 1]

def minimum_weight_matching(G, odd_vert):
    """Perform minimum weight matching using Blossom algorithm."""
    subgraph = nx.Graph()
    # Use positive weights for compatibility with max_weight_matching
    for v in odd_vert:
        for u in odd_vert:
            if u != v:
                subgraph.add_edge(v, u, weight=G[v][u])
    matching = nx.max_weight_matching(subgraph, maxcardinality=True)
    # Convert matching to list of edges with original weights
    return [(u, v, G[u][v]) for u, v in matching]

def find_eulerian_tour(MST, G):
    """Find an Eulerian tour in the augmented graph."""
    neighbors = {v: [] for v in G}
    for u, v, _ in MST:
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
    """Convert an Eulerian tour into a Hamiltonian circuit."""
    path, visited = [], set()
    for v in eulerian_tour:
        if v not in visited:
            path.append(v)
            visited.add(v)
    path.append(path[0])  # Complete the circuit
    length = sum(G[path[i]][path[i + 1]] for i in range(len(path) - 1))
    return length, path
