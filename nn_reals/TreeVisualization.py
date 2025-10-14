import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import networkx as nx
from collections import defaultdict

class ClusterNode:
    """Node in hierarchical clustering tree"""
    def __init__(self, node_id, networks, parent=None, distance_threshold=0):
        self.node_id = node_id
        self.networks = networks
        self.children = []
        self.parent = parent
        self.distance_threshold = distance_threshold
    
    def avg_fitness(self):
        return np.mean([net.fitness() for net in self.networks]) if self.networks else 0
    
    def size(self):
        return len(self.networks)

# Hierarchical clustering tree with decreasing distance thresholds
# generation_label = label for the generation to visualize.
# thresholds = List of decreasing distance thresholds [d0, d1, d2, ...], If None, will auto-generate one using autothresholds
class HierarchicalClusteringTree:
    def __init__(self, networks, generation_label=None, metric='euclidean', 
                 thresholds=None, auto_thresholds=5):
        self.networks = networks
        self.generation_label = generation_label
        self.metric = metric
        self.root = None
        self.all_nodes = []
        self._node_counter = 0
        
        # Compute distances first
        self.distances = self._compute_distances()
        
        # Set up thresholds
        if thresholds is not None:
            # Validate that thresholds are decreasing
            if not all(thresholds[i] > thresholds[i+1] for i in range(len(thresholds)-1)):
                raise ValueError("Thresholds must be strictly decreasing")
            self.thresholds = thresholds
        else:
            # Auto-generate decreasing thresholds from distance distribution
            self.thresholds = self._auto_generate_thresholds(auto_thresholds)
    
    # Create tree from Population object's current generation
    @classmethod
    def from_population(cls, population, **kwargs):
        return cls(networks=population.pop, generation_label="Current Generation", **kwargs)
    
    # Create tree from specific generation in Neuroevolution history
    @classmethod
    def from_neuroevolution(cls, neuroevolution, generation, **kwargs):
        if generation >= len(neuroevolution.pop_history):
            raise ValueError(f"Generation {generation} not found. History has "
                           f"{len(neuroevolution.pop_history)} generations")
        return cls(networks=neuroevolution.pop_history[generation], 
                  generation_label=f"Generation {generation}", **kwargs)
    
    def _compute_distances(self):
        """Compute pairwise genetic distances"""
        from nn_reals.Population import Population
        n = len(self.networks)
        dist = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                d = Population.genetic_distance(self.networks[i], self.networks[j], 
                                               metric=self.metric)
                dist[i, j] = dist[j, i] = d
        return dist
    
    # Auto-generate decreasing thresholds from distance distribution
    def _auto_generate_thresholds(self, n_levels):
        # Get all unique pairwise distances
        unique_dists = np.unique(self.distances[np.triu_indices_from(self.distances, k=1)])
        
        if len(unique_dists) == 0:
            return [0] * n_levels
        
        # Use decreasing percentiles: start high, go low
        percentiles = np.linspace(95, 10, n_levels)
        thresholds = [np.percentile(unique_dists, p) for p in percentiles]
        
        print(f"Auto-generated thresholds: {[f'{t:.4f}' for t in thresholds]}")
        return thresholds
    
    def build_tree(self):
        """Build the hierarchical clustering tree"""
        print(f"Building tree for {self.generation_label or 'networks'} "
              f"({len(self.thresholds)} levels)")
        print(f"Thresholds: {[f'{t:.4f}' for t in self.thresholds]}")
        
        # Start with all networks at root
        self.root = self._build_recursive(list(range(len(self.networks))), 0)
        
        print(f"Tree built with {len(self.all_nodes)} nodes")
        return self.root
    
    def _build_recursive(self, indices, level):
        """Recursively build tree by partitioning at distance thresholds"""
        networks = [self.networks[i] for i in indices]
        
        # Get threshold for this level
        threshold = self.thresholds[level] if level < len(self.thresholds) else 0
        
        # Create node
        node = ClusterNode(self._node_counter, networks, distance_threshold=threshold)
        self.all_nodes.append(node)
        self._node_counter += 1
        
        # Stop if we're at max depth or have a single network
        if level >= len(self.thresholds) - 1 or len(indices) <= 1:
            return node
        
        # Partition networks into subclusters where all pairs are within next threshold
        next_threshold = self.thresholds[level + 1]
        clusters = self._partition_at_threshold(indices, next_threshold)
        
        # Create children for each subcluster
        if len(clusters) > 1:
            for cluster in clusters:
                if cluster:  # Non-empty cluster
                    child = self._build_recursive(cluster, level + 1)
                    child.parent = node
                    node.children.append(child)
        
        return node
    
    def _partition_at_threshold(self, indices, threshold):
        """
        Partition indices into clusters where *all pairwise distances* <= threshold.
        Uses complete-linkage clustering rule.
        """
        if len(indices) <= 1:
            return [indices]

        clusters = [[i] for i in indices]  # start with each point as its own cluster
        merged = True

        while merged:
            merged = False
            new_clusters = []
            skip = set()

            for i in range(len(clusters)):
                if i in skip:
                    continue
                current_cluster = clusters[i]
                for j in range(i + 1, len(clusters)):
                    if j in skip:
                        continue
                    candidate_cluster = clusters[j]

                    # compute the maximum distance between any pair across clusters
                    max_dist = max(
                        self.distances[a, b] for a in current_cluster for b in candidate_cluster
                    )

                    # if all pairwise distances ≤ threshold, merge them
                    if max_dist <= threshold:
                        current_cluster = current_cluster + candidate_cluster
                        skip.add(j)
                        merged = True

                new_clusters.append(current_cluster)
            clusters = new_clusters

        return clusters

    # Verify that all clusters satisfy the distance constraints
    def verify_clusters(self):
        def verify_node(node, level):
            if len(node.networks) <= 1:
                return True
            
            threshold = self.thresholds[level] if level < len(self.thresholds) else 0
            
            # Get indices of networks in this node
            indices = [i for i, net in enumerate(self.networks) if net in node.networks]
            
            # Check all pairwise distances
            max_dist = 0
            violations = 0
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    dist = self.distances[indices[i], indices[j]]
                    max_dist = max(max_dist, dist)
                    if dist > threshold:
                        violations += 1
            
            print(f"Level {level}, Node {node.node_id}: "
                  f"size={node.size()}, threshold={threshold:.4f}, "
                  f"max_dist={max_dist:.4f}, violations={violations}")
            
            # Verify children
            for child in node.children:
                verify_node(child, level + 1)
        
        print("\n=== Verifying cluster constraints ===")
        verify_node(self.root, 0)
    
    def visualize(self, figsize=(16, 10), save_path=None):
        """Visualize the hierarchical clustering tree"""
        if self.root is None:
            raise ValueError("Tree not built. Call build_tree() first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Build graph
        G = nx.DiGraph()
        pos = {}
        self._add_to_graph(self.root, G, pos, 0, 0, 0, 4)
        
        # Color by fitness
        fitnesses = [node.avg_fitness() for node in self.all_nodes]
        f_min, f_max = min(fitnesses), max(fitnesses)
        norm_fit = [(f - f_min) / (f_max - f_min) if f_max != f_min else 0.5 
                    for f in fitnesses]
        
        cmap = LinearSegmentedColormap.from_list('fitness', ['green', 'yellow', 'red'])
        colors = [cmap(f) for f in norm_fit]
        sizes = [300 + node.size() * 50 for node in self.all_nodes]
        
        # Draw
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', arrows=True, 
                              arrowsize=15, width=1.5)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=sizes, 
                              alpha=0.8, edgecolors='black', linewidths=1.5)
        
        labels = {n.node_id: f"{n.size()}\n{n.avg_fitness():.3f}" for n in self.all_nodes}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8)
        
        # Title and colorbar
        title = f"Hierarchical Clustering Tree (Decreasing Distance Thresholds)"
        if self.generation_label:
            title += f" - {self.generation_label}"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=f_min, vmax=f_max))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Average Fitness (lower is better)', rotation=270, labelpad=20)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        plt.show()
    
    def _add_to_graph(self, node, G, pos, x, y, layer, dx_base):
        """Recursively add nodes to graph"""
        G.add_node(node.node_id)
        pos[node.node_id] = (x, -layer)
        
        if node.children:
            n = len(node.children)
            dx = dx_base / (layer + 1) if layer > 0 else dx_base
            x_start = x - (n - 1) * dx / 2
            
            for i, child in enumerate(node.children):
                G.add_edge(node.node_id, child.node_id)
                self._add_to_graph(child, G, pos, x_start + i * dx, y, layer + 1, dx_base)
    
    def print_structure(self, node=None, depth=0):
        """Print tree structure for debugging"""
        if node is None:
            node = self.root
        print(f"{'  ' * depth}Node {node.node_id}: size={node.size()}, "
              f"fitness={node.avg_fitness():.4f}, threshold={node.distance_threshold:.4f}")
        for child in node.children:
            self.print_structure(child, depth + 1)