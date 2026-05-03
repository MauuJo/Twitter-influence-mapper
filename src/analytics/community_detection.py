import pickle
import pandas as pd
import networkx as nx
import community.community_louvain as community_louvain
import os


class CommunityDetector:
    def __init__(self, graph_path="data/graphs/twitter_graph.gpickle"):
        self.graph_path = graph_path

    def load_graph(self):
        print("Loading graph...")

        with open(self.graph_path, "rb") as f:
            G = pickle.load(f)

        print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def detect_communities(self, G):
        print("Running Louvain community detection...")

        # Louvain requires undirected graph
        undirected_G = G.to_undirected()

        partition = community_louvain.best_partition(undirected_G, weight="weight")

        print(f"Detected {len(set(partition.values()))} communities")

        return partition

    def save_communities(self, partition):
        df = pd.DataFrame({
            "username": list(partition.keys()),
            "community_id": list(partition.values())
        })

        output_path = "data/processed/community_labels.csv"
        df.to_csv(output_path, index=False)

        print(f"Community labels saved to {output_path}")

    def display_top_communities(self, partition):
        community_counts = pd.Series(partition).value_counts()

        print("\nTop communities by size:\n")
        print(community_counts.head(10))

    def run_pipeline(self):
        G = self.load_graph()

        partition = self.detect_communities(G)

        self.save_communities(partition)

        self.display_top_communities(partition)

        print("Community detection pipeline completed successfully.")


if __name__ == "__main__":
    detector = CommunityDetector()
    detector.run_pipeline()