import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from pyvis.network import Network
import tempfile
import os


# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Twitter Influence Intelligence Platform",
    layout="wide"
)

st.title("🐦 Twitter Social Influence Intelligence Platform")
st.markdown("Analyze influential users, hidden communities, and information ecosystems across Twitter news discussions.")


# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    insights_df = pd.read_csv("data/processed/final_social_insights.csv", low_memory=False)
    summary_df = pd.read_csv("data/processed/executive_summary.csv")
    topics_df = pd.read_csv("data/processed/community_topics.csv")
    strategic_df = pd.read_csv("data/processed/strategic_insights.csv")
    community_sentiment_df = pd.read_csv("data/processed/community_sentiment.csv")

    # Cast community_id to same type across all dataframes
    community_sentiment_df["community_id"] = community_sentiment_df["community_id"].astype(int)
    topics_df["community_id"] = topics_df["community_id"].astype(int)
    insights_df["community_id"] = pd.to_numeric(insights_df["community_id"], errors="coerce").dropna().astype(int)

    # Build a clean community_id -> label map from insights_df (one row per community)
    community_label_map = (
        insights_df[["community_id", "community_label"]]
        .dropna(subset=["community_label"])
        .drop_duplicates(subset="community_id")
    )

    # Build a clean community_id -> keywords map from topics_df
    topics_clean = (
        topics_df[["community_id", "top_keywords"]]
        .dropna(subset=["top_keywords"])
        .drop_duplicates(subset="community_id")
    )

    # Merge both
    community_sentiment_df = community_sentiment_df.merge(
        community_label_map,
        on="community_id",
        how="left"
    )
    community_sentiment_df = community_sentiment_df.merge(
        topics_clean,
        on="community_id",
        how="left"
    )

    # For communities too small to get a topic label, fall back to community_id as label
    community_sentiment_df["community_label"] = community_sentiment_df["community_label"].fillna(
        "Community " + community_sentiment_df["community_id"].astype(str)
    )
    community_sentiment_df["top_keywords"] = community_sentiment_df["top_keywords"].fillna("< 10 tweets — no keywords")
    influencer_sentiment_df = pd.read_csv("data/processed/influencer_sentiment.csv")

    with open("data/graphs/twitter_graph.gpickle", "rb") as f:
        G = pickle.load(f)

    return insights_df, summary_df, topics_df, strategic_df, community_sentiment_df, influencer_sentiment_df, G


insights_df, summary_df, topics_df, strategic_df, community_sentiment_df, influencer_sentiment_df, G = load_data()


# ---------------------------
# SIDEBAR NAVIGATION
# ---------------------------
page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Dashboard",
        "Influencer Intelligence",
        "Community Ecosystems",
        "User Intelligence Lookup",
        "Strategic Insights",
        "Sentiment Intelligence",
        "Network Visualization"
    ]
)


# ---------------------------
# EXECUTIVE DASHBOARD
# ---------------------------
if page == "Executive Dashboard":
    st.header("📊 Executive Network Overview")

    summary = summary_df.iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Top Influencer", summary["Top Influencer"])
    col2.metric("Largest Community", summary["Largest Community"])
    col3.metric("Elite Influencers", summary["Total Elite Influencers"])
    col4.metric("Verified Elite Influencers", summary["Verified Influencers"])

    st.subheader("Influence Tier Distribution")

    tier_counts = insights_df["influence_tier"].value_counts().reset_index()
    tier_counts.columns = ["Tier", "Count"]

    fig = px.pie(
        tier_counts,
        names="Tier",
        values="Count",
        title="Influence Tier Breakdown"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 10 Influencers")

    top_users = insights_df.sort_values("pagerank", ascending=False).head(10)

    st.dataframe(
        top_users[
            [
                "username",
                "influence_tier",
                "user_role",
                "community_label",
                "followersCount",
                "verified"
            ]
        ],
        use_container_width=True
    )


# ---------------------------
# INFLUENCER INTELLIGENCE
# ---------------------------
elif page == "Influencer Intelligence":
    st.header("🏆 Influencer Intelligence Center")

    tier_filter = st.selectbox(
        "Filter by Influence Tier",
        insights_df["influence_tier"].unique()
    )

    filtered_df = insights_df[
        insights_df["influence_tier"] == tier_filter
    ]

    st.dataframe(
        filtered_df.sort_values("pagerank", ascending=False),
        use_container_width=True
    )


# ---------------------------
# COMMUNITY ECOSYSTEMS
# ---------------------------
elif page == "Community Ecosystems":
    st.header("🏘️ Community Ecosystem Explorer")

    community_labels = insights_df["community_label"].dropna().unique()

    selected_label = st.selectbox(
        "Select Community Type",
        sorted(community_labels)
    )

    community_df = insights_df[
        insights_df["community_label"] == selected_label
    ]

    st.subheader(f"{selected_label} Overview")

    st.metric("Community Size", len(community_df))

    st.subheader("Top Community Members")

    st.dataframe(
        community_df.sort_values("pagerank", ascending=False).head(20),
        use_container_width=True
    )

    st.subheader("Community Role Distribution")

    role_counts = community_df["user_role"].value_counts().reset_index()
    role_counts.columns = ["Role", "Count"]

    fig = px.bar(
        role_counts,
        x="Role",
        y="Count",
        title=f"{selected_label} User Roles"
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------
# USER LOOKUP
# ---------------------------
elif page == "User Intelligence Lookup":
    st.header("🔍 User Intelligence Lookup")

    username = st.text_input("Enter Twitter Username")

    if username:
        username = username.lower().strip().replace("@", "")

        user_df = insights_df[
            insights_df["username"].str.lower() == username
        ]

        if not user_df.empty:
            user = user_df.iloc[0]

            st.subheader(f"Profile: @{user['username']}")

            st.success(
                f"{user['username']} is a {user['influence_tier']} functioning as a "
                f"{user['user_role']} within the {user['community_label']} ecosystem."
            )

            col1, col2, col3 = st.columns(3)

            col1.metric("Followers", user.get("followersCount", "N/A"))
            col2.metric("Verified", user.get("verified", "N/A"))
            col3.metric("Community", user["community_label"])

            st.dataframe(user_df, use_container_width=True)

        else:
            st.warning("User not found in network.")


# ---------------------------
# NETWORK VISUALIZATION
# ---------------------------
elif page == "Network Visualization":
    st.header("🌐 Social Network Visualization")

    st.info("Showing top 300 influential users for performance.")

    top_nodes = insights_df.sort_values(
        "pagerank",
        ascending=False
    ).head(300)["username"].tolist()

    subgraph = G.subgraph(top_nodes)

    net = Network(height="800px", width="100%", directed=True)

    color_map = {
        "Elite Influencer": "red",
        "Key Amplifier": "orange",
        "Active Contributor": "blue",
        "Peripheral Participant": "gray"
    }

    for node in subgraph.nodes():
        user_data = insights_df[
            insights_df["username"] == node
        ]

        if user_data.empty:
            continue

        user = user_data.iloc[0]

        net.add_node(
            node,
            label=node,
            title=f"""
            Role: {user['user_role']}
            Tier: {user['influence_tier']}
            Community: {user['community_label']}
            """,
            size=10 + user["pagerank"] * 5000,
            color=color_map.get(user["influence_tier"], "gray")
        )

    for source, target, data in subgraph.edges(data=True):
        net.add_edge(
            source,
            target,
            value=data.get("weight", 1)
        )

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    temp_file.close()

    net.save_graph(temp_file.name)

    with open(temp_file.name, "r", encoding="utf-8") as f:
        html_content = f.read()

    st.components.v1.html(html_content, height=850)

elif page == "Strategic Insights":
    st.header("🧠 Why This Network Matters")

    strategic = strategic_df.iloc[0]

    col1, col2 = st.columns(2)

    col1.metric(
        "Most Influential Verified User",
        strategic["Most Influential Verified User"]
    )

    col2.metric(
        "Most Influential Non-Verified User",
        strategic["Most Influential Non-Verified User"]
    )

    col3, col4 = st.columns(2)

    col3.metric(
        "Largest Community Ecosystem",
        strategic["Largest Community Ecosystem"]
    )

    col4.metric(
        "Potential High-Risk Narrative Cluster",
        strategic["Potential High-Risk Narrative Cluster"]
    )

    st.subheader("🔗 Hidden Bridge Accounts")
    st.write(strategic["Top Bridge Accounts"])

    st.subheader("📈 Verified vs Non-Verified Influence Comparison")

    comparison_df = pd.DataFrame({
        "Category": ["Verified", "Non-Verified"],
        "Average Influence": [
            strategic["Avg Verified Influence"],
            strategic["Avg Non-Verified Influence"]
        ]
    })

    fig = px.bar(
        comparison_df,
        x="Category",
        y="Average Influence",
        title="Influence Comparison"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success(
        f"""
        Verified accounts demonstrate stronger average influence,
        while {strategic['Largest Community Ecosystem']} represents
        the dominant ecosystem shaping discourse.
        """
    )

elif page == "Sentiment Intelligence":
    st.header("💬 Sentiment & Narrative Intelligence")

    # Overall sentiment distribution
    st.subheader("Community Sentiment Distribution")

    sentiment_counts = community_sentiment_df["community_sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    fig = px.pie(
        sentiment_counts,
        names="Sentiment",
        values="Count",
        title="Overall Community Sentiment"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Most negative communities
    st.subheader("⚠️ Most Negative Communities")

    MIN_TWEETS = 10

    filtered_sentiment_df = community_sentiment_df[
        community_sentiment_df["tweet_count"] >= MIN_TWEETS
    ]

    negative_communities = filtered_sentiment_df.sort_values("sentiment_score").head(10)

    st.dataframe(
        negative_communities[
            [
                "community_label",
                "top_keywords",
                "sentiment_score",
                "community_sentiment",
                "tweet_count"
            ]
        ],
        use_container_width=True
    )

    # Most positive communities
    st.subheader("🌟 Most Positive Communities")

    positive_communities = filtered_sentiment_df.sort_values("sentiment_score", ascending=False).head(10)

    st.dataframe(
        positive_communities[
            [
                "community_label",
                "top_keywords",
                "sentiment_score",
                "community_sentiment",
                "tweet_count"
            ]
        ],
        use_container_width=True
    )

    # Influencer sentiment
    st.subheader("🧠 Influencer Sentiment Overview")

    insights_df["username"] = insights_df["username"].astype(str).str.lower().str.strip().str.replace("@", "", regex=False)

    influencer_sentiment_df["username"] = influencer_sentiment_df["username"].astype(str).str.lower().str.strip().str.replace("@", "", regex=False)

    # FIX: deduplicate insights before merge, use NaN not 0 for missing sentiment
    insights_deduped = insights_df.drop_duplicates(subset="username").copy()
    insights_deduped["username"] = insights_deduped["username"].astype(str).str.lower().str.strip().str.replace("@", "", regex=False)

    influencer_sentiment_df["username"] = influencer_sentiment_df["username"].astype(str).str.lower().str.strip().str.replace("@", "", regex=False)

    influencer_merge = insights_deduped.merge(
        influencer_sentiment_df[["username", "sentiment_score"]],
        on="username",
        how="left"
    )

    def sentiment_category(score):
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        else:
            return "Neutral"
    # Don't fillna(0) — NaN means no tweets found, not neutral sentiment
    influencer_merge["sentiment_label"] = influencer_merge["sentiment_score"].apply(
        lambda score: sentiment_category(score) if pd.notna(score) else "No Data"
    )
    

    

    top_sentiment_users = influencer_merge.sort_values(
        "pagerank",
        ascending=False
    ).head(20)

    st.dataframe(
        top_sentiment_users[
            [
                "username",
                "influence_tier",
                "user_role",
                "community_label",
                "sentiment_label",
                "sentiment_score"
            ]
        ],
        use_container_width=True
    )

    st.success(
        """
        Sentiment analysis reveals emotional patterns within major communities,
        helping identify polarized clusters, positive ecosystems,
        and high-risk negative discourse zones.
        """
    )