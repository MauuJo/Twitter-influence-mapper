import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os


class TopicLabeler:
    def __init__(self, processed_path="data/processed"):
        self.tweets_file = os.path.join(processed_path, "cleaned_tweets.csv")
        self.community_file = os.path.join(processed_path, "community_labels.csv")

    def load_data(self):
        print("Loading tweets and community labels...")

        tweets_df = pd.read_csv(self.tweets_file)
        communities_df = pd.read_csv(self.community_file)

        print(f"Tweets: {tweets_df.shape}")
        print(f"Communities: {communities_df.shape}")

        return tweets_df, communities_df

    def merge_data(self, tweets_df, communities_df):
        print("Normalizing usernames...")

        tweets_df["username"] = tweets_df["username"].astype(str).str.lower().str.strip().str.replace("@", "", regex=False)
        communities_df["username"] = communities_df["username"].astype(str).str.lower().str.strip().str.replace("@", "", regex=False)

        print("Merging tweets with communities...")

        merged_df = tweets_df.merge(
            communities_df,
            on="username",
            how="inner"
        )

        print(f"Merged dataset: {merged_df.shape}")

        return merged_df

    def extract_topics(self, merged_df, top_n=5):
        print("Extracting community topics...")

        community_topics = []

        grouped = merged_df.groupby("community_id")

        for community_id, group in grouped:
            texts = group["content"].dropna().astype(str).tolist()

            if len(texts) < 10:
                continue

            combined_text = " ".join(texts)

            stop_words = list(TfidfVectorizer(stop_words="english").get_stop_words())
            custom_stopwords = stop_words + ["https", "news", "00", "amp", "co"]
            vectorizer = TfidfVectorizer(
                stop_words=custom_stopwords,
                max_features=1000,
                token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]{2,}\b'
            )

            try:
                tfidf_matrix = vectorizer.fit_transform([combined_text])
                feature_names = vectorizer.get_feature_names_out()

                scores = tfidf_matrix.toarray()[0]

                top_keywords = [
                    feature_names[i]
                    for i in scores.argsort()[-top_n:][::-1]
                ]

                community_topics.append({
                    "community_id": community_id,
                    "top_keywords": ", ".join(top_keywords),
                    "size": len(group)
                })

            except:
                continue

        topics_df = pd.DataFrame(community_topics)

        return topics_df

    def save_topics(self, topics_df):
        output_path = "data/processed/community_topics.csv"
        topics_df.to_csv(output_path, index=False)

        print(f"Community topics saved to {output_path}")

    def display_top_topics(self, topics_df):
        print("\nTop community topics:\n")
        print(topics_df.sort_values(by="size", ascending=False).head(10))

    def run_pipeline(self):
        tweets_df, communities_df = self.load_data()

        merged_df = self.merge_data(tweets_df, communities_df)

        topics_df = self.extract_topics(merged_df)

        self.save_topics(topics_df)

        self.display_top_topics(topics_df)

        print("Topic labeling pipeline completed successfully.")


if __name__ == "__main__":
    labeler = TopicLabeler()
    labeler.run_pipeline()