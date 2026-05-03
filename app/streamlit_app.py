import streamlit as st
import pandas as pd
import networkx as nx
import pickle
import plotly.express as px
from pyvis.network import Network
import tempfile
import os


# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Twitter Influence Mapper",
    layout="wide"
)

st.title("🐦 Twitter Social Network Influence Mapper")


# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    metrics_df = pd.read_csv("data/processed/influence_metrics.csv")
    communities_df = pd.read_csv("data/processed/community_labels.csv")
    topics_df = pd.read_csv("data/processed/community_topics.csv")
    users_df = pd.read_csv("data/processed/cleaned_users.csv")

    with open("data/graphs/twitter_graph.gpickle", "rb") as f:
        G = pickle.load(f)

    return metrics_df, communities_df, topics_df, users_df, G


metrics_df, communities_df, topics_df, users_df, G = load_data()


# ---------------------------
# SIDEBAR NAVIGATION
# ---------------------------
page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard Overview",
        "Influence Leaderboard",
        "Community Explorer",
        "User Lookup",
        "Network Visualization"
    ]
)


# ---------------------------
# DASHBOARD OVERVIEW
# ---------------------------
if page == "Dashboard Overview":
    st.header("📊 Network Statistics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Nodes", G.number_of_nodes())
    col2.metric("Total Edges", G.number_of_edges())
    col3.metric("Communities", communities_df["community_id"].nunique())
    col4.metric(
        "Top Influencer",
        metrics_df.sort_values("pagerank", ascending=False).iloc[0]["username"]
    )

    st.subheader("Top 10 Users by PageRank")

    top_users = metrics_df.sort_values("pagerank", ascending=False).head(10)

    fig = px.bar(
        top_users,
        x="username",
        y="pagerank",
        title="Top Influencers"
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------
# INFLUENCE LEADERBOARD
# ---------------------------
elif page == "Influence Leaderboard":
    st.header("🏆 Influence Leaderboard")

    metric = st.selectbox(
        "Select Metric",
        [
            "pagerank",
            "degree_centrality",
            "betweenness_centrality",
            "eigenvector_centrality"
        ]
    )

    leaderboard = metrics_df.sort_values(metric, ascending=False).head(50)

    st.dataframe(
        leaderboard[["username", metric]],
        use_container_width=True
    )


# ---------------------------
# COMMUNITY EXPLORER
# ---------------------------
elif page == "Community Explorer":
    st.header("🏘️ Community Explorer")

    community_ids = sorted(topics_df["community_id"].unique())

    selected_community = st.selectbox(
        "Select Community ID",
        community_ids
    )

    topic_info = topics_df[topics_df["community_id"] == selected_community]

    st.subheader("Community Topic Keywords")

    if not topic_info.empty:
        st.write(topic_info.iloc[0]["top_keywords"])
        st.write(f"Community Size: {topic_info.iloc[0]['size']}")

    members = communities_df[
        communities_df["community_id"] == selected_community
    ]

    merged_members = members.merge(metrics_df, on="username")

    st.subheader("Top Community Members")

    st.dataframe(
        merged_members.sort_values("pagerank", ascending=False).head(20),
        use_container_width=True
    )


# ---------------------------
# USER LOOKUP
# ---------------------------
elif page == "User Lookup":
    st.header("🔍 User Lookup")

    username = st.text_input("Enter Twitter Username")

    if username:
        username = username.lower().strip().replace("@", "")

        user_metrics = metrics_df[
            metrics_df["username"].str.lower() == username
        ]

        user_community = communities_df[
            communities_df["username"].str.lower() == username
        ]

        user_profile = users_df[
            users_df["username"].str.lower() == username
        ]

        if not user_metrics.empty:
            st.subheader("Influence Metrics")
            st.dataframe(user_metrics, use_container_width=True)

            if not user_community.empty:
                st.write(f"Community ID: {user_community.iloc[0]['community_id']}")

            if not user_profile.empty:
                st.subheader("Profile Metadata")
                st.dataframe(user_profile, use_container_width=True)

        else:
            st.warning("User not found.")


# ---------------------------
# NETWORK VISUALIZATION
# ---------------------------
elif page == "Network Visualization":
    st.header("🌐 Network Graph")

    st.info("Displaying top 300 users by PageRank for performance.")

    top_nodes = metrics_df.sort_values(
        "pagerank",
        ascending=False
    ).head(300)["username"].tolist()

    subgraph = G.subgraph(top_nodes)

    net = Network(height="750px", width="100%", directed=True)

    for node in subgraph.nodes():
        pagerank_score = metrics_df[
            metrics_df["username"] == node
        ]["pagerank"].values[0]

        net.add_node(
            node,
            label=node,
            size=10 + pagerank_score * 5000
        )

    for source, target, data in subgraph.edges(data=True):
        net.add_edge(
            source,
            target,
            value=data.get("weight", 1)
        )

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(temp_file.name)

    with open(temp_file.name, "r", encoding="utf-8") as f:
        html_content = f.read()

    st.components.v1.html(html_content, height=800)

    os.unlink(temp_file.name)