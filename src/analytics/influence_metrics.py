import pickle
import pandas as pd
import networkx as nx
import os


class InfluenceMetrics:
    def __init__(self, graph_path="data/graphs/twitter_graph.gpickle"):
        self.graph_path = graph_path

    def load_graph(self):
        print("Loading graph...")

        with open(self.graph_path, "rb") as f:
            G = pickle.load(f)

        print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def compute_metrics(self, G):
        print("Computing PageRank...")
        pagerank = nx.pagerank(G, weight="weight")

        print("Computing Degree Centrality...")
        degree = nx.degree_centrality(G)

        print("Computing Betweenness Centrality...")
        betweenness = nx.betweenness_centrality(G, weight="weight")

        print("Computing Eigenvector Centrality...")
        try:
            eigenvector = nx.eigenvector_centrality_numpy(G, weight="weight")
        except:
            eigenvector = {node: 0 for node in G.nodes()}

        metrics_df = pd.DataFrame({
            "username": list(G.nodes()),
            "pagerank": [pagerank[node] for node in G.nodes()],
            "degree_centrality": [degree[node] for node in G.nodes()],
            "betweenness_centrality": [betweenness[node] for node in G.nodes()],
            "eigenvector_centrality": [eigenvector[node] for node in G.nodes()]
        })

        return metrics_df

    def save_metrics(self, df):
        output_path = "data/processed/influence_metrics.csv"
        df.to_csv(output_path, index=False)

        print(f"Metrics saved to {output_path}")

    def display_top_users(self, df, metric="pagerank", top_n=10):
        print(f"\nTop {top_n} users by {metric}:\n")
        print(df.sort_values(by=metric, ascending=False)[["username", metric]].head(top_n))

    def run_pipeline(self):
        G = self.load_graph()

        metrics_df = self.compute_metrics(G)

        self.save_metrics(metrics_df)

        self.display_top_users(metrics_df, "pagerank")
        self.display_top_users(metrics_df, "betweenness_centrality")

        print("Influence metrics pipeline completed successfully.")


if __name__ == "__main__":
    analyzer = InfluenceMetrics()
    analyzer.run_pipeline()