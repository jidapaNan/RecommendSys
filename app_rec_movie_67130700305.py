# app.py — Streamlit interface for movie recommendations

from pathlib import Path
from typing import Tuple, Optional, List, Union
import importlib
import pickle

import streamlit as st
import pandas as pd
import numpy as np

# ---------- Page config & debug ----------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")
st.set_option("client.showErrorDetails", True)

st.title("🎬 Movie Recommender (User-based)")
st.write("Use your precomputed similarity + ratings to get movie recommendations for a selected user.")

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def load_pickle_bytes(file_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame]:
    obj = pickle.loads(file_bytes)
    if not isinstance(obj, (tuple, list)) or len(obj) != 2:
        raise ValueError("Pickle must contain a tuple/list: (user_similarity_df, user_movie_ratings)")
    user_similarity_df, user_movie_ratings = obj
    if not isinstance(user_similarity_df, pd.DataFrame):
        raise TypeError("user_similarity_df must be a pandas DataFrame")
    if not isinstance(user_movie_ratings, pd.DataFrame):
        raise TypeError("user_movie_ratings must be a pandas DataFrame")
    return user_similarity_df, user_movie_ratings

@st.cache_data(show_spinner=False)
def load_pickle_path(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    with path.open("rb") as f:
        return load_pickle_bytes(f.read())

@st.cache_resource(show_spinner=False)
def import_recommender(module_name: str = "myfunction_67130700305",
                       func_name: str = "get_movie_recommendations"):
    mod = importlib.import_module(module_name)
    if not hasattr(mod, func_name):
        raise AttributeError(f"Function '{func_name}' not found in module '{module_name}'.")
    return getattr(mod, func_name)

def candidate_user_ids(sim_df: Optional[pd.DataFrame],
                       ratings_df: Optional[pd.DataFrame]) -> List[Union[int, str]]:
    """Find user IDs from index/columns of provided dataframes (robust to int/str)."""
    ids: List[Union[int, str]] = []

    def _coerce(v):
        try:
            return int(v)
        except Exception:
            return str(v)

    if sim_df is not None:
        if len(sim_df.index) > 0:
            ids += [_coerce(i) for i in sim_df.index.tolist()]
        if len(sim_df.columns) > 0:
            ids += [_coerce(i) for i in sim_df.columns.tolist()]
    if ratings_df is not None:
        if len(ratings_df.index) > 0:
            ids += [_coerce(i) for i in ratings_df.index.tolist()]
        if len(ratings_df.columns) > 0:
            ids += [_coerce(i) for i in ratings_df.columns.tolist()]

    # de-duplicate while preserving order
    seen, dedup = set(), []
    for i in ids:
        k = (type(i), i)
        if k not in seen:
            seen.add(k)
            dedup.append(i)
    return dedup

# Try to import recommendation function (non-fatal if missing)
try:
    get_movie_recommendations = import_recommender()
except Exception as e:
    st.warning("⚠️ Could not import your recommendation function yet. "
               "You can still load data and configure settings below.\n\n"
               f"Import error: `{e}`")
    get_movie_recommendations = None

# ---------- 1) Load data ----------
st.subheader("1) Load your data")

colA, colB = st.columns([1, 1])
with colA:
    use_uploader = st.toggle("Upload .pkl file", value=False,
                             help="Toggle to upload a pickle file instead of using a local path.")
with colB:
    default_path = st.text_input("Local pickle path",
                                 value="recommendation_data.pkl",
                                 help="If not uploading, the app will try to load this path.")

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

# ---------- 2) Preview + parameters ----------
if user_similarity_df is not None and user_movie_ratings is not None:
    with st.expander("Preview dataframes (first 10 rows)"):
        st.markdown("**user_similarity_df**")
        st.dataframe(user_similarity_df.head(10), use_container_width=True)
        st.markdown("**user_movie_ratings**")
        st.dataframe(user_movie_ratings.head(10), use_container_width=True)

    st.subheader("2) Pick a user and parameters")

    user_ids = candidate_user_ids(user_similarity_df, user_movie_ratings)
    if not user_ids:
        st.error("ไม่พบรายชื่อผู้ใช้จากไฟล์ที่โหลดมา — "
                 "ตรวจสอบว่า similarity/ratings มี index หรือ columns เป็น user ID")
        st.stop()

    c1, c2 = st.columns([1, 1])
    with c1:
        user_id = st.selectbox("User ID", options=user_ids, index=0)
    with c2:
        k = st.slider("How many recommendations?", min_value=1, max_value=50, value=10, step=1)

    # ---------- 3) Recommend ----------
    st.subheader("3) Get recommendations")
    if st.button("🚀 Recommend"):
        if get_movie_recommendations is None:
            st.error("Could not import `get_movie_recommendations`. "
                     "Ensure `myfunction_67130700305.py` exists in the repo.")
        else:
            try:
                recs = get_movie_recommendations(user_id, user_similarity_df, user_movie_ratings, k)
                if recs is None:
                    st.warning("No recommendations returned.")
                else:
                    # normalize to list of strings
                    if isinstance(recs, (pd.Series, pd.Index)):
                        recs = recs.tolist()
                    elif isinstance(recs, pd.DataFrame) and recs.shape[1] == 1:
                        recs = recs.iloc[:, 0].tolist()
                    elif not isinstance(recs, (list, tuple)):
                        recs = [str(recs)]
                    df_out = pd.DataFrame({"Rank": range(1, len(recs) + 1),
                                           "Movie": [str(x) for x in recs]})
                    st.success(f"Top {len(recs)} movie recommendations for user {user_id}")
                    st.dataframe(df_out, use_container_width=True)
            except Exception as e:
                st.error("🚨 Exception while generating recommendations:")
                st.exception(e)

    # ---------- Optional: show top similar users ----------
    with st.expander("Show top similar users for the selected user"):
        try:
            sims_series = None
            if user_id in user_similarity_df.index:
                sims_series = user_similarity_df.loc[user_id]
            elif str(user_id) in user_similarity_df.index:
                sims_series = user_similarity_df.loc[str(user_id)]
            elif user_id in user_similarity_df.columns:
                sims_series = user_similarity_df[user_id]
            elif str(user_id) in user_similarity_df.columns:
                sims_series = user_similarity_df[str(user_id)]

            if sims_series is not None:
                sims = sims_series.sort_values(ascending=False)
                sims = sims.drop(labels=[user_id, str(user_id)], errors="ignore").head(10)
                st.dataframe(
                    sims.reset_index().rename(
                        columns={sims.index.name or "index": "User", 0: "Similarity"}
                    ),
                    use_container_width=True
                )
            else:
                st.info("Selected user not found in similarity matrix (index/columns).")
        except Exception as e:
            st.info(f"Similarity view unavailable: {e}")
else:
    st.stop()

st.caption("Built with Streamlit • Need export to CSV/Excel or diagnostics? Ping me ✨")
