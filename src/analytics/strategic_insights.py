import pandas as pd
import os


class StrategicInsights:
    def __init__(self, processed_path="data/processed"):
        self.insights_file = os.path.join(processed_path, "final_social_insights.csv")

    def load_data(self):
        df = pd.read_csv(self.insights_file, low_memory=False)
        return df

    def generate_insights(self, df):
        insights = {}

        # Most influential verified user
        verified_users = df[df["verified"] == True]
        if not verified_users.empty:
            top_verified = verified_users.sort_values("pagerank", ascending=False).iloc[0]
            insights["Most Influential Verified User"] = top_verified["username"]

        # Most influential non-verified user
        non_verified_users = df[df["verified"] != True]
        if not non_verified_users.empty:
            top_non_verified = non_verified_users.sort_values("pagerank", ascending=False).iloc[0]
            insights["Most Influential Non-Verified User"] = top_non_verified["username"]

        # Largest ecosystem
        largest_community = df["community_label"].value_counts().idxmax()
        insights["Largest Community Ecosystem"] = largest_community

        # Hidden bridge accounts
        bridge_users = df.sort_values("betweenness_centrality", ascending=False).head(5)
        insights["Top Bridge Accounts"] = ", ".join(bridge_users["username"].tolist())

        # Verified vs Non-Verified influence
        verified_avg = verified_users["pagerank"].mean() if not verified_users.empty else 0
        non_verified_avg = non_verified_users["pagerank"].mean() if not non_verified_users.empty else 0

        insights["Avg Verified Influence"] = round(verified_avg, 6)
        insights["Avg Non-Verified Influence"] = round(non_verified_avg, 6)

        # Potential misinformation cluster
        suspicious_clusters = df[
            df["community_label"].str.contains(
                "Conservative|Political|Media",
                case=False,
                na=False
            )
        ]

        if not suspicious_clusters.empty:
            dominant_cluster = suspicious_clusters["community_label"].value_counts().idxmax()
            insights["Potential High-Risk Narrative Cluster"] = dominant_cluster

        return insights

    def save_insights(self, insights):
        output_path = "data/processed/strategic_insights.csv"

        pd.DataFrame([insights]).to_csv(output_path, index=False)

        print(f"Strategic insights saved to {output_path}")

    def display_insights(self, insights):
        print("\nStrategic Insights:\n")

        for key, value in insights.items():
            print(f"{key}: {value}")

    def run_pipeline(self):
        df = self.load_data()

        insights = self.generate_insights(df)

        self.save_insights(insights)

        self.display_insights(insights)

        print("\nStrategic insight generation completed successfully.")


if __name__ == "__main__":
    analyzer = StrategicInsights()
    analyzer.run_pipeline()