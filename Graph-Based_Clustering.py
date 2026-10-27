# EXPERIMENT 7: GRAPH-BASED CLUSTERING
#
# Aim:
# To implement Graph-Based Clustering for grouping related objects.
#
# Theory:
# Graph-Based Clustering represents objects as nodes and relationships as edges.
# A Minimum Spanning Tree (MST) connects all nodes with minimum total edge weight.
# Removing high-weight edges can separate the graph into clusters.
#
# Algorithm:
# 1. Import NetworkX and Matplotlib.
# 2. Create a weighted graph.
# 3. Calculate the Minimum Spanning Tree.
# 4. Remove edges above a selected weight threshold.
# 5. Identify clusters using connected components.
# 6. Display clusters and visualize the graph.
#
# Output:
# Displays the identified clusters and graph visualization.
#
# Result:
# Graph-Based Clustering is demonstrated using an MST.

import networkx as nx
import matplotlib.pyplot as plt

# Create a weighted graph
G = nx.Graph()

edges = [
    (0, 1, 1),
    (1, 2, 1),
    (2, 3, 5),
    (3, 4, 1),
    (4, 5, 1)
]

G.add_weighted_edges_from(edges)

# Calculate Minimum Spanning Tree
mst = nx.minimum_spanning_tree(G)

# Remove high-weight edges to create clusters
threshold = 2
cluster_graph = mst.copy()

for u, v, data in list(cluster_graph.edges(data=True)):
    if data["weight"] > threshold:
        cluster_graph.remove_edge(u, v)

# Find clusters
clusters = list(nx.connected_components(cluster_graph))

print("Clusters:")
for i, cluster in enumerate(clusters, start=1):
    print("Cluster", i, ":", sorted(cluster))

# Visualize the clustered graph
pos = nx.spring_layout(cluster_graph, seed=42)

nx.draw(
    cluster_graph,
    pos,
    with_labels=True,
    node_color="lightblue",
    node_size=800
)

nx.draw_networkx_edge_labels(
    cluster_graph,
    pos,
    edge_labels=nx.get_edge_attributes(cluster_graph, "weight")
)

plt.title("Graph-Based Clustering using MST")
plt.show()

# Result: The graph is divided into clusters by removing high-weight edges.