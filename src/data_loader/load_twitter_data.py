import pandas as pd
import os


class TwitterDataLoader:
    def __init__(self, raw_data_path="data/raw"):
        self.raw_data_path = raw_data_path

        self.network_file = os.path.join(raw_data_path, "weighted_twitter_news_network.csv")
        self.tweets_file = os.path.join(raw_data_path, "news_tweets.csv")
        self.users_file = os.path.join(raw_data_path, "news_tweeters.csv")

        for file in [self.network_file, self.tweets_file, self.users_file]:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Missing file: {file}")

    def load_network_data(self):
        print("Loading network data...")
        df = pd.read_csv(self.network_file)

        print(f"Network data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print("Columns:", df.columns.tolist())

        return df

    def load_tweets_data(self):
        print("Loading tweets data...")
        df = pd.read_csv(
            self.tweets_file,
            engine="python",
            on_bad_lines="skip",
            encoding="utf-8",
            quoting=3
        )

        print(f"Tweets data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print("Columns:", df.columns.tolist())

        return df

    def load_users_data(self):
        print("Loading user data...")
        df = pd.read_csv(
            self.users_file,
            engine="python",
            on_bad_lines="skip",
            encoding="utf-8"
        )

        print(f"Users data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print("Columns:", df.columns.tolist())

        return df

    def clean_network_data(self, df):
        print("Cleaning network data...")

        # Drop missing rows
        df = df.dropna()

        # Remove self-loops
        if "source" in df.columns and "target" in df.columns:
            df = df[df["source"] != df["target"]]

        # Standardize column names
        df.columns = [col.strip().lower() for col in df.columns]

        print(f"Cleaned network data: {df.shape[0]} rows")

        return df

    def save_processed_data(self, df, filename):
        output_path = os.path.join("data/processed", filename)
        df.to_csv(output_path, index=False)

        print(f"Saved processed data to {output_path}")

    def run_full_pipeline(self):
        network_df = self.load_network_data()
        cleaned_network_df = self.clean_network_data(network_df)

        tweets_df = self.load_tweets_data()
        users_df = self.load_users_data()

        self.save_processed_data(cleaned_network_df, "cleaned_network.csv")
        self.save_processed_data(tweets_df, "cleaned_tweets.csv")
        self.save_processed_data(users_df, "cleaned_users.csv")

        print("Data loading and preprocessing completed successfully.")


if __name__ == "__main__":
    loader = TwitterDataLoader()
    loader.run_full_pipeline()