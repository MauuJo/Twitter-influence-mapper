import pandas as pd
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon")


class SentimentAnalyzer:
    def __init__(self, processed_path="data/processed"):
        self.tweets_file = os.path.join(processed_path, "cleaned_tweets.csv")
        self.communities_file = os.path.join(processed_path, "community_labels.csv")
        self.final_insights_file = os.path.join(processed_path, "final_social_insights.csv")

        self.sia = SentimentIntensityAnalyzer()

    def load_data(self):
        tweets_df = pd.read_csv(self.tweets_file, low_memory=False)
        communities_df = pd.read_csv(self.communities_file)
        insights_df = pd.read_csv(self.final_insights_file, low_memory=False)

        return tweets_df, communities_df, insights_df

    def normalize_usernames(self, df):
        df["username"] = df["username"].astype(str).str.lower().str.strip().str.replace("@", "", regex=False)
        return df

    def compute_sentiment(self, text):
        if pd.isna(text):
            return 0

        score = self.sia.polarity_scores(str(text))["compound"]
        return score

    def analyze_tweets(self, tweets_df):
        print("Computing sentiment scores for tweets...")

        tweets_df["sentiment_score"] = tweets_df["content"].apply(self.compute_sentiment)

        def classify(score):
            if score >= 0.05:
                return "Positive"
            elif score <= -0.05:
                return "Negative"
            else:
                return "Neutral"

        tweets_df["sentiment_label"] = tweets_df["sentiment_score"].apply(classify)

        return tweets_df

    def merge_with_communities(self, tweets_df, communities_df):
        tweets_df = self.normalize_usernames(tweets_df)
        communities_df = self.normalize_usernames(communities_df)

        merged_df = tweets_df.merge(
            communities_df,
            on="username",
            how="inner"
        )

        return merged_df

    def community_sentiment_summary(self, merged_df):
        print("Generating community sentiment summaries...")

        summary = merged_df.groupby("community_id").agg(
            sentiment_score=("sentiment_score", "mean"),
            tweet_count=("username", "count")
        ).reset_index()

        def classify(score):
            if score >= 0.05:
                return "Positive"
            elif score <= -0.05:
                return "Negative"
            else:
                return "Neutral"

        summary["community_sentiment"] = summary["sentiment_score"].apply(classify)
        return summary
    def influencer_sentiment_summary(self, merged_df):
        influencer_summary = merged_df.groupby("username").agg({
            "sentiment_score": "mean",
            "content": "count"
        }).reset_index()

        influencer_summary.rename(columns={"content": "tweet_count"}, inplace=True)

        return influencer_summary

    def save_outputs(self, tweets_df, community_summary, influencer_summary):
        tweets_df.to_csv("data/processed/tweet_sentiment.csv", index=False)
        community_summary.to_csv("data/processed/community_sentiment.csv", index=False)
        influencer_summary.to_csv("data/processed/influencer_sentiment.csv", index=False)

        print("Sentiment outputs saved successfully.")

    def run_pipeline(self):
        tweets_df, communities_df, insights_df = self.load_data()

        tweets_df = self.analyze_tweets(tweets_df)

        merged_df = self.merge_with_communities(tweets_df, communities_df)

        community_summary = self.community_sentiment_summary(merged_df)

        influencer_summary = self.influencer_sentiment_summary(merged_df)

        self.save_outputs(
            tweets_df,
            community_summary,
            influencer_summary
        )

        print("\nTop sentiment communities:")
        print(community_summary.head())

        print("\nSentiment analysis completed successfully.")


if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    analyzer.run_pipeline()