import os
import io
import json
import time
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
from scipy import sparse

# Project imports
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.preprocessing import DataLoader
from src.models import SemanticHybridRecommender
from src.metrics import mapk_score

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception as e:
    st.warning("sentence-transformers is required for semantic features. Install via requirements.txt")

APP_TITLE = "Library Recommender — Semantic Hybrid"
DATA_DIR_DEFAULT = os.path.join(PROJECT_ROOT, 'data')
SUBMISSION_DIR_DEFAULT = os.path.join(PROJECT_ROOT, 'submission')
FEEDBACK_LOG = os.path.join(DATA_DIR_DEFAULT, 'feedback_log.csv')

st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")

# -----------------------------
# Caching helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data(path_interactions: str, path_items: str) -> Tuple[DataLoader, pd.DataFrame, pd.DataFrame]:
    loader = DataLoader(path_interactions, path_items)
    return loader, loader.interactions, loader.items_df

@st.cache_resource(show_spinner=True)
def build_model_cached(interactions: pd.DataFrame,
                       items_df: pd.DataFrame,
                       n_users: int,
                       n_items: int,
                       alpha: float,
                       half_life_days: List[int],
                       ensemble_weights: List[float] | None) -> SemanticHybridRecommender:
    model = SemanticHybridRecommender(n_users=n_users, n_items=n_items)
    model.fit(interactions, items_df, alpha=alpha, half_life_days=half_life_days, ensemble_weights=ensemble_weights)
    return model

@st.cache_resource(show_spinner=False)
def load_bert() -> SentenceTransformer:
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data(show_spinner=False)
def compute_item_embeddings(items_df: pd.DataFrame) -> np.ndarray:
    model_bert = load_bert()
    df_items_sorted = items_df.sort_values('i_idx').fillna('')
    texts = (df_items_sorted.get('Title', '').astype(str) + '. ' +
             df_items_sorted.get('Author', '').astype(str) + '. ' +
             df_items_sorted.get('Subjects', '').astype(str)).tolist()
    return model_bert.encode(texts, show_progress_bar=False)

# -----------------------------
# Utilities
# -----------------------------
def topk_similar_items(item_sim_matrix: np.ndarray, i_idx: int, k: int = 10) -> List[int]:
    row = item_sim_matrix[i_idx]
    # Avoid self (set to -inf)
    row = row.copy()
    if i_idx < len(row):
        row[i_idx] = -1e9
    idx = np.argpartition(row, -k)[-k:]
    return idx[np.argsort(row[idx])[::-1]].tolist()


def build_item_similarity_from_model(model: SemanticHybridRecommender) -> np.ndarray:
    # Average item_matrix across ensemble models using their weights
    mats = []
    weights = []
    for m in model.ensemble_models:
        mats.append(m['item_matrix'])
        weights.append(m['weight'])
    if not mats:
        return np.zeros((model.n_items, model.n_items), dtype=float)
    # Some may be sparse
    acc = None
    for w, mat in zip(weights, mats):
        if sparse.issparse(mat):
            mat = mat.toarray()
        if acc is None:
            acc = w * mat
        else:
            acc += w * mat
    return acc


def recommend_for_user(model: SemanticHybridRecommender, u_idx: int, k: int = 10) -> List[int]:
    preds = model.predict(k=k, batch_size=max(1000, k))
    return preds[u_idx].tolist()


def fallback_popular(interactions: pd.DataFrame, k: int = 10) -> List[int]:
    top = interactions['i_idx'].value_counts().head(k).index.tolist()
    return top


def save_feedback(log_path: str, payload: dict):
    try:
        df = pd.DataFrame([payload])
        if os.path.exists(log_path):
            df.to_csv(log_path, mode='a', header=False, index=False)
        else:
            df.to_csv(log_path, index=False)
    except Exception as e:
        st.error(f"Failed to save feedback: {e}")

# -----------------------------
# Sidebar — configuration
# -----------------------------
st.sidebar.header("Configuration")
path_inter = st.sidebar.text_input("Path — interactions_train.csv", os.path.join(DATA_DIR_DEFAULT, 'interactions_train.csv'))
use_enriched = st.sidebar.toggle("Use enriched items (LLM)", value=True)
path_items = st.sidebar.text_input(
    "Path — items.csv",
    os.path.join(DATA_DIR_DEFAULT, 'items_enriched_ai_turbo.csv' if use_enriched else 'items.csv')
)

alpha = st.sidebar.slider("Alpha (collab vs content)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
hl_str = st.sidebar.text_input("Half-life days (comma-separated)", value="1,250")
try:
    half_life_days = [int(x.strip()) for x in hl_str.split(',') if x.strip()]
    if len(half_life_days) == 0:
        half_life_days = [1, 250]
except Exception:
    half_life_days = [1, 250]

k_top = st.sidebar.slider("Top-K", min_value=5, max_value=50, value=10, step=1)
refit = st.sidebar.button("Rebuild model (fit)")
clear_caches = st.sidebar.button("Clear caches")

if clear_caches:
    load_data.clear()
    build_model_cached.clear()
    compute_item_embeddings.clear()
    st.sidebar.success("Caches cleared.")

# -----------------------------
# Load data
# -----------------------------
st.title(APP_TITLE)

with st.spinner("Loading data..."):
    try:
        loader, interactions_df, items_df = load_data(path_inter, path_items)
        st.success(f"Loaded {len(interactions_df):,} interactions, {len(items_df):,} items.")
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

# -----------------------------
# Fit / load model
# -----------------------------
model_key = json.dumps({
    'alpha': alpha,
    'hl': half_life_days,
    'n_users': loader.n_users,
    'n_items': loader.n_items,
    'paths': [os.path.abspath(path_inter), os.path.abspath(path_items)]
})

if refit:
    build_model_cached.clear()

with st.spinner("Fitting model (first time may take a few minutes for BERT)..."):
    model = build_model_cached(
        interactions=loader.get_full_data(),
        items_df=items_df,
        n_users=loader.n_users,
        n_items=loader.n_items,
        alpha=alpha,
        half_life_days=half_life_days,
        ensemble_weights=None
    )

# Precompute helpers
item_sim_matrix = build_item_similarity_from_model(model)

# For cold-start semantic search
try:
    item_emb = compute_item_embeddings(items_df)
    item_emb_norm = item_emb / (np.linalg.norm(item_emb, axis=1, keepdims=True) + 1e-12)
except Exception:
    item_emb = None
    item_emb_norm = None

# -----------------------------
# Tabs / Pages
# -----------------------------
main_tab, user_tab, cold_tab, similar_tab, analytics_tab, settings_tab = st.tabs([
    "Overview", "Recommend for Patron", "Cold-Start (Text Search)", "Similar Books", "Analytics", "Settings"
])

with main_tab:
    st.subheader("Welcome 👋")
    st.markdown(
        """
        This app showcases the final Semantic Hybrid recommender used in your academic project:
        - Time-decayed collaborative profiles + S-BERT content similarity
        - Decoupled re-buy boost and light popularity prior
        - Tunable `alpha` and half-lives to balance recent vs long-term preferences
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Users", f"{loader.n_users:,}")
    with col2:
        st.metric("Items", f"{loader.n_items:,}")
    with col3:
        st.metric("Interactions", f"{len(interactions_df):,}")

    st.markdown("---")
    st.markdown("### Quick catalog search")
    q = st.text_input("Title / Author / Subject contains", "")
    n_show = st.slider("Max rows", 5, 200, 20)
    if q:
        ql = q.lower()
        def contains(s):
            try:
                return ql in str(s).lower()
            except Exception:
                return False
        mask = items_df['Title'].apply(contains) | items_df.get('Author', '').apply(contains) | items_df.get('Subjects', '').apply(contains)
        st.dataframe(items_df.loc[mask, ['i', 'Title', 'Author', 'Subjects']].head(n_show), use_container_width=True)
    else:
        st.dataframe(items_df[['i', 'Title', 'Author', 'Subjects']].head(n_show), use_container_width=True)

with user_tab:
    st.subheader("Recommendations for a known patron")
    user_id = st.selectbox("Select user_id", options=loader.u_unique)
    u_idx = loader.u_map[user_id]

    exclude_read = st.checkbox("Exclude already borrowed (may lower MAP@K)", value=False)
    max_per_author = st.number_input("Max per author in Top-K (0 for no limit)", min_value=0, max_value=10, value=0, step=1)

    if st.button("Recommend", type='primary'):
        with st.spinner("Scoring..."):
            indices = recommend_for_user(model, u_idx, k=k_top)
            recs = items_df.set_index('i_idx').loc[indices].copy()
            recs['score_rank'] = range(1, len(recs) + 1)

            # Apply business rules
            if exclude_read:
                history_items = interactions_df[interactions_df['u_idx'] == u_idx]['i_idx'].unique().tolist()
                recs = recs[~recs.index.isin(history_items)]

            if max_per_author > 0 and 'Author' in recs.columns:
                recs = recs.groupby('Author', group_keys=False).head(max_per_author)

        st.write("Top recommendations:")
        st.dataframe(recs.reset_index()[['score_rank', 'i', 'Title', 'Author', 'Subjects']].head(k_top), use_container_width=True)

        # Download
        csv_buf = io.StringIO()
        out_df = recs.reset_index()[['i', 'Title', 'Author', 'Subjects']].head(k_top)
        out_df.to_csv(csv_buf, index=False)
        st.download_button("Download CSV", data=csv_buf.getvalue(), file_name=f"recs_user_{user_id}.csv", mime='text/csv')

        # Feedback
        st.markdown("### Feedback")
        fb_cols = st.columns(len(recs.head(k_top)))
        for col, (idx, row) in zip(fb_cols, recs.head(k_top).iterrows()):
            with col:
                liked = st.toggle(f"👍 {row.get('Title', 'Item')}\n({row.get('Author', '')})", key=f"fb_{user_id}_{int(idx)}")
                if liked:
                    save_feedback(FEEDBACK_LOG, {
                        'ts': int(time.time()),
                        'user_id': user_id,
                        'i': int(row.get('i', -1)),
                        'i_idx': int(idx),
                        'title': row.get('Title', ''),
                        'author': row.get('Author', ''),
                        'like': 1
                    })

with cold_tab:
    st.subheader("Cold-start patron: describe interests")
    st.caption("Enter keywords, a short description, or paste a book title and author.")
    query = st.text_area("Interests", placeholder="e.g., space exploration, Asimov, robots, classic sci-fi")
    if st.button("Find similar books"):
        if item_emb_norm is None:
            st.error("Semantic encoder not available. Install sentence-transformers.")
        else:
            with st.spinner("Encoding query and ranking..."):
                bert = load_bert()
                q_vec = bert.encode([query], show_progress_bar=False)
                q_vec = q_vec / (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-12)
                sims = (q_vec @ item_emb_norm.T).ravel()
                idx = np.argpartition(sims, -k_top)[-k_top:]
                idx = idx[np.argsort(sims[idx])[::-1]]
                recs = items_df.set_index('i_idx').loc[idx]
                st.dataframe(recs.reset_index()[['i', 'Title', 'Author', 'Subjects']].head(k_top), use_container_width=True)

with similar_tab:
    st.subheader("Similar books (item-item)")
    # Select by title substring to pick an item
    title_q = st.text_input("Type title to search", "")
    candidates = items_df[items_df['Title'].str.contains(title_q, case=False, na=False)].head(50)
    if len(candidates) == 0:
        st.info("No matching titles yet.")
    else:
        picked = st.selectbox("Pick a book", options=list(candidates['Title'] + '  —  #' + candidates['i'].astype(str)))
        # parse item id at the end
        try:
            picked_i = int(picked.split('#')[-1])
            i_idx = int(items_df[items_df['i'] == picked_i]['i_idx'].iloc[0])
            neighbors = topk_similar_items(item_sim_matrix, i_idx, k=k_top)
            recs = items_df.set_index('i_idx').loc[neighbors]
            st.dataframe(recs.reset_index()[['i', 'Title', 'Author', 'Subjects']], use_container_width=True)
        except Exception as e:
            st.error(f"Failed to compute similar items: {e}")

with analytics_tab:
    st.subheader("Collection & usage snapshots")
    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("Unique patrons", f"{loader.n_users:,}")
    with colB:
        st.metric("Unique books", f"{loader.n_items:,}")
    with colC:
        top_pop = fallback_popular(interactions_df, k=5)
        st.metric("Top popular (i_idx)", ", ".join(map(str, top_pop)))

    st.markdown("#### Top authors by interactions")
    if 'Author' in items_df.columns:
        merged = interactions_df.merge(items_df[['i', 'i_idx', 'Author']], on=['i', 'i_idx'], how='left')
        top_auth = merged['Author'].fillna('Unknown').value_counts().head(20)
        st.bar_chart(top_auth)
    else:
        st.info("Author column not available in items.")

    st.markdown("#### Recency histogram (per-user rank pct)")
    if 'rank' not in interactions_df.columns:
        tmp = interactions_df.sort_values(['u_idx', 't']).copy()
        interactions_df = interactions_df.copy()
        interactions_df['rank'] = tmp.groupby('u_idx')['t'].rank(pct=True, method='dense')
    # Draw histogram using numpy and st.bar_chart
    hist_counts, bin_edges = np.histogram(interactions_df['rank'].astype(float).fillna(0.0), bins=30, range=(0.0, 1.0))
    hist_df = pd.DataFrame({
        'bin': pd.IntervalIndex.from_breaks(bin_edges).astype(str),
        'count': hist_counts
    })
    hist_df = hist_df.set_index('bin')
    st.bar_chart(hist_df)

with settings_tab:
    st.subheader("Settings & Utilities")
    st.write("Paths:")
    st.code(f"interactions: {path_inter}\nitems: {path_items}")

    st.write("Export sample submission (warm users only)")
    if st.button("Generate CSV preview"):
        with st.spinner("Predicting for all known users (preview)..."):
            preds = model.predict(k=k_top, batch_size=1000)
            out = []
            for uid in loader.u_unique[:1000]:  # limit preview
                u_idx = loader.u_map[uid]
                pred_items = [str(loader.idx_to_i[i]) for i in preds[u_idx]]
                out.append({'user_id': uid, 'recommendation': ' '.join(pred_items)})
            df_out = pd.DataFrame(out)
            csv_buf = io.StringIO()
            df_out.to_csv(csv_buf, index=False)
            st.download_button("Download preview CSV", data=csv_buf.getvalue(), file_name="submission_preview.csv", mime='text/csv')

st.caption("© 2025 Library Recommender Demo — Streamlit UI. This demo reads local CSVs; see README for details.")
