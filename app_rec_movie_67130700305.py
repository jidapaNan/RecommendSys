# app.py — Streamlit interface for movie recommendations
# ----------------------------------------------------
# How it works
# - Loads a pickled tuple: (user_similarity_df, user_movie_ratings)
# - Imports your function: get_movie_recommendations(user_id, user_similarity_df, user_movie_ratings, k)
# - Lets you pick a user and number of recommendations, then shows results
#
# Usage
#   1) pip install streamlit pandas numpy
#   2) Put recommendation_data.pkl and myfunction_67130700305.py in the same folder
#   3) streamlit run app.py

from pathlib import Path
import pickle
import importlib
from typing import Tuple, Optional, List

import streamlit as st
import pandas as pd
import numpy as np

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

st.title("🎬 Movie Recommender (User-based)")
st.write("Use your precomputed similarity + ratings to get movie recommendations for a selected user.")

# ----------------------------
# Utilities
# ----------------------------
@st.cache_data(show_spinner=False)
def load_pickle_bytes(file_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame]:
    obj = pickle.loads(file_bytes)
    if not isinstance(obj, (tuple, list)) or len(obj) != 2:
        raise ValueError("Pickle must contain a tuple/list: (user_similarity_df, user_movie_ratings)")
    user_similarity_df, user_movie_ratings = obj
    # Basic sanity checks
    if not isinstance(user_similarity_df, pd.DataFrame):
        raise TypeError("user_similarity_df must be a pandas DataFrame")
    if not isinstance(user_movie_ratings, pd.DataFrame):
        raise TypeError("user_movie_ratings must be a pandas DataFrame")
    return user_similarity_df, user_movie_ratings

@st.cache_data(show_spinner=False)
def load_pickle_path(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    with path.open('rb') as f:
        return load_pickle_bytes(f.read())

@st.cache_resource(show_spinner=False)
def import_recommender(module_name: str = "myfunction_67130700305", func_name: str = "get_movie_recommendations"):
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        raise ImportError(f"Could not import module '{module_name}'. Make sure the file {module_name}.py exists.") from e
    if not hasattr(mod, func_name):
        raise AttributeError(f"Function '{func_name}' not found in module '{module_name}'.")
    return getattr(mod, func_name)

# Try to import the function once
try:
    get_movie_recommendations = import_recommender()
except Exception as e:
    st.warning("⚠️ Could not import your recommendation function yet. You can still load data and configure settings below.")
    get_movie_recommendations = None

# ----------------------------
# Data loading UI
# ----------------------------
st.subheader("1) Load your data")

colA, colB = st.columns([1, 1])

with colA:
    use_uploader = st.toggle("Upload .pkl file", value=False, help="Toggle to upload a pickle file instead of using a local path.")

with colB:
    default_path = st.text_input("Local pickle path", value="recommendation_data.pkl", help="If not uploading, the app will try to load this path.")

user_similarity_df: Optional[pd.DataFrame] = None
user_movie_ratings: Optional[pd.DataFrame] = None

if use_uploader:
    up = st.file_uploader("Upload recommendation_data.pkl", type=["pkl", "pickle"], accept_multiple_files=False)
    if up is not None:
        try:
            user_similarity_df, user_movie_ratings = load_pickle_bytes(up.getvalue())
            st.success("✅ Data loaded from uploaded file.")
        except Exception as e:
            st.error(f"Failed to load pickle: {e}")
else:
    path = Path(default_path)
    if path.exists():
        try:
            user_similarity_df, user_movie_ratings = load_pickle_path(path)
            st.success(f"✅ Data loaded from: {path}")
        except Exception as e:
            st.error(f"Failed to load pickle: {e}")
    else:
        st.info("ℹ️ Waiting for data… Upload a file or fix the local path.")

# ----------------------------
# Preview + Controls
# ----------------------------
if user_similarity_df is not None and user_movie_ratings is not None:
    with st.expander("Preview dataframes (first 10 rows)"):
        st.markdown("**user_similarity_df**")
        st.dataframe(user_similarity_df.head(10))
        st.markdown("**user_movie_ratings**")
        st.dataframe(user_movie_ratings.head(10))

    st.subheader("2) Pick a user and parameters")
    # Derive candidate user IDs from the index/columns heuristics
    def candidate_user_ids(sim_df: pd.DataFrame, ratings_df: pd.DataFrame) -> List[int]:
        def _to_int_list(idx) -> List[int]:
            # แปลงค่าใน index เป็น int อย่างปลอดภัย
            try:
                return pd.Index(idx).astype("int64", errors="ignore").tolist()
            except Exception:
                out = []
                for v in idx:
                    try:
                        out.append(int(v))
                    except Exception:
                        pass
                return out

        # 1) ลองจาก index ของ similarity matrix ก่อน
        ids = _to_int_list(sim_df.index)

        # 2) ถ้ายังว่าง ลองจาก index ของ ratings
        if not ids:
            ids = _to_int_list(ratings_df.index)

        # 3) ถ้ายังไม่ได้จริง ๆ ให้ทำเป็น 1..N
        if not ids:
            ids = list(range(1, len(ratings_df) + 1))

        # unique + sort
        try:
            ids = sorted(set(int(i) for i in ids))
        except Exception:
            pass
        return ids

        # ใช้เหมือนเดิม
        user_ids = candidate_user_ids(user_similarity_df, user_movie_ratings)

    c1, c2 = st.columns([1,1])
    with c1:
        user_id = st.selectbox("User ID", options=user_ids, index=0 if user_ids else None)
    with c2:
        k = st.slider("How many recommendations?", min_value=1, max_value=50, value=10, step=1)

    st.subheader("3) Get recommendations")
    run = st.button("🚀 Recommend")

    if run:
        if get_movie_recommendations is None:
            st.error("Could not import get_movie_recommendations. Ensure myfunction_67130700305.py is present.")
        else:
            try:
                recs = get_movie_recommendations(user_id, user_similarity_df, user_movie_ratings, k)
                if recs is None:
                    st.warning("No recommendations returned.")
                else:
                    # Normalize to list of strings
                    if isinstance(recs, (pd.Series, pd.Index)):
                        recs = recs.tolist()
                    elif isinstance(recs, pd.DataFrame) and recs.shape[1] == 1:
                        recs = recs.iloc[:, 0].tolist()
                    elif not isinstance(recs, (list, tuple)):
                        recs = [str(recs)]

                    df_out = pd.DataFrame({"Rank": list(range(1, len(recs)+1)), "Movie": [str(x) for x in recs]})
                    st.success(f"Top {len(recs)} movie recommendations for user {user_id}")
                    st.dataframe(df_out, use_container_width=True)
            except Exception as e:
                st.exception(e)

    # Optional insights
    with st.expander("Show top similar users for the selected user"):
        try:
            if user_id in user_similarity_df.index:
                sims = user_similarity_df.loc[user_id].sort_values(ascending=False).drop(labels=[user_id], errors='ignore').head(10)
                st.dataframe(sims.reset_index().rename(columns={sims.index.name or 'index': 'User', 0: 'Similarity'}))
            else:
                st.info("Selected user not found in similarity matrix index.")
        except Exception as e:
            st.info("Similarity view unavailable.")

else:
    st.stop()

st.caption("Built with Streamlit • If you need export to CSV/Excel or model diagnostics, let me know ✨")


