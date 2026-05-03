import pandas as pd
import os


class InsightEngine:
    def __init__(self, processed_path="data/processed"):
        self.metrics_file = os.path.join(processed_path, "influence_metrics.csv")
        self.communities_file = os.path.join(processed_path, "community_labels.csv")
        self.topics_file = os.path.join(processed_path, "community_topics.csv")
        self.users_file = os.path.join(processed_path, "cleaned_users.csv")

    def load_data(self):
        metrics_df = pd.read_csv(self.metrics_file)
        communities_df = pd.read_csv(self.communities_file)
        topics_df = pd.read_csv(self.topics_file)
        users_df = pd.read_csv(self.users_file, low_memory=False)

        return metrics_df, communities_df, topics_df, users_df

    def assign_influence_tier(self, metrics_df):
        pagerank_thresholds = metrics_df["pagerank"].quantile([0.99, 0.95, 0.80])

        def classify(score):
            if score >= pagerank_thresholds[0.99]:
                return "Elite Influencer"
            elif score >= pagerank_thresholds[0.95]:
                return "Key Amplifier"
            elif score >= pagerank_thresholds[0.80]:
                return "Active Contributor"
            else:
                return "Peripheral Participant"

        metrics_df["influence_tier"] = metrics_df["pagerank"].apply(classify)

        return metrics_df

    def assign_user_roles(self, metrics_df):
        def role(row):
            if row["betweenness_centrality"] > metrics_df["betweenness_centrality"].quantile(0.95):
                return "Community Bridge"
            elif row["degree_centrality"] > metrics_df["degree_centrality"].quantile(0.95):
                return "Network Hub"
            elif row["pagerank"] > metrics_df["pagerank"].quantile(0.95):
                return "Information Broadcaster"
            else:
                return "Regular Participant"

        metrics_df["user_role"] = metrics_df.apply(role, axis=1)

        return metrics_df

    def label_communities(self, topics_df):
        def interpret_keywords(keywords):
            text = keywords.lower()

            if any(word in text for word in ["trump", "fox", "gop", "conservative"]):
                return "Conservative Political Media"
            elif any(word in text for word in ["biden", "cnn", "democrat", "liberal"]):
                return "Liberal Political Media"
            elif any(word in text for word in ["crypto", "bitcoin", "finance"]):
                return "Finance / Crypto"
            elif any(word in text for word in ["sports", "nba", "football"]):
                return "Sports Community"
            elif any(word in text for word in ["news", "media"]):
                return "General News Community"
            else:
                return "Mixed Interest Community"

        topics_df["community_label"] = topics_df["top_keywords"].apply(interpret_keywords)

        return topics_df

    def merge_all_data(self, metrics_df, communities_df, topics_df, users_df):
        merged = metrics_df.merge(communities_df, on="username", how="left")
        merged = merged.merge(users_df, on="username", how="left")
        merged = merged.merge(
            topics_df[["community_id", "community_label"]],
            on="community_id",
            how="left"
        )

        return merged

    def generate_summary(self, merged_df):
        top_influencer = merged_df.sort_values("pagerank", ascending=False).iloc[0]
        largest_community = merged_df["community_label"].value_counts().idxmax()

        insights = {
            "Top Influencer": top_influencer["username"],
            "Top Influencer Role": top_influencer["user_role"],
            "Largest Community": largest_community,
            "Total Elite Influencers": (merged_df["influence_tier"] == "Elite Influencer").sum(),
            "Verified Influencers": merged_df[
                (merged_df["verified"] == True) &
                (merged_df["influence_tier"] == "Elite Influencer")
            ].shape[0]
        }

        return insights

    def save_outputs(self, merged_df, insights):
        merged_df.to_csv("data/processed/final_social_insights.csv", index=False)

        pd.DataFrame([insights]).to_csv(
            "data/processed/executive_summary.csv",
            index=False
        )

        print("Saved final insights and executive summary.")

    def run_pipeline(self):
        metrics_df, communities_df, topics_df, users_df = self.load_data()

        metrics_df = self.assign_influence_tier(metrics_df)
        metrics_df = self.assign_user_roles(metrics_df)

        topics_df = self.label_communities(topics_df)

        merged_df = self.merge_all_data(
            metrics_df,
            communities_df,
            topics_df,
            users_df
        )

        insights = self.generate_summary(merged_df)

        self.save_outputs(merged_df, insights)

        print("\nExecutive Summary:")
        for key, value in insights.items():
            print(f"{key}: {value}")

        print("\nInsight engine completed successfully.")


if __name__ == "__main__":
    engine = InsightEngine()
    engine.run_pipeline()