import pandas as pd
import networkx as nx
import os
import pickle


class TwitterGraphBuilder:
    def __init__(self, processed_data_path="data/processed"):
        self.network_file = os.path.join(processed_data_path, "cleaned_network.csv")

    def load_network_data(self):
        print("Loading processed network data...")
        df = pd.read_csv(self.network_file)

        print(f"Loaded {df.shape[0]} interactions")
        return df

    def build_graph(self, df):
        print("Building directed weighted graph...")

        G = nx.DiGraph()

        for _, row in df.iterrows():
            source = row["source"]
            target = row["target"]
            weight = row["weight"]

            if G.has_edge(source, target):
                G[source][target]["weight"] += weight
            else:
                G.add_edge(source, target, weight=weight)

        print(f"Graph built successfully:")
        print(f"Nodes: {G.number_of_nodes()}")
        print(f"Edges: {G.number_of_edges()}")

        return G

    def filter_graph(self, G, min_degree=2):
        print(f"Filtering nodes with degree < {min_degree}...")

        nodes_to_remove = [
            node for node, degree in dict(G.degree()).items()
            if degree < min_degree
        ]

        G.remove_nodes_from(nodes_to_remove)

        print("Filtered graph stats:")
        print(f"Nodes: {G.number_of_nodes()}")
        print(f"Edges: {G.number_of_edges()}")

        return G

    def save_graph(self, G):
        output_path = "data/graphs/twitter_graph.gpickle"
        with open(output_path, "wb") as f:
            pickle.dump(G, f)

        print(f"Graph saved to {output_path}")

    def run_pipeline(self):
        df = self.load_network_data()

        G = self.build_graph(df)

        G = self.filter_graph(G, min_degree=2)

        self.save_graph(G)

        print("Graph pipeline completed successfully.")


if __name__ == "__main__":
    builder = TwitterGraphBuilder()
    builder.run_pipeline()