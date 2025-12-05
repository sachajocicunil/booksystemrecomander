"""
📚 BiblioRec — Système de Recommandation Intelligent pour Bibliothèques
Application Streamlit pour démontrer le modèle Super-Ensemble
"""

import os
import io
import json
import time
import hashlib
from typing import List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import sparse

# Project imports
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.preprocessing import DataLoader
from src.models import SemanticHybridRecommender

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# =============================================================================
# CONFIGURATION
# =============================================================================
APP_TITLE = "📚 BiblioRec — Recommandations Intelligentes"
APP_ICON = "📚"
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
FEEDBACK_LOG = os.path.join(DATA_DIR, 'feedback_log.csv')

# Page config
st.set_page_config(
    page_title="BiblioRec",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS pour un look moderne
st.markdown("""
<style>
    /* Header gradient */
    .main-header {
        background: linear-gradient(90deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .metric-card p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 0.9rem;
    }
    
    /* Book cards */
    .book-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .book-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .book-title {
        font-weight: bold;
        color: #1e3a5f;
        font-size: 1.1rem;
    }
    .book-author {
        color: #666;
        font-style: italic;
    }
    .book-reason {
        background: #f0f7ff;
        padding: 0.5rem;
        border-radius: 5px;
        font-size: 0.85rem;
        margin-top: 0.5rem;
        color: #2d5a87;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
    }
    
    /* Success/Info boxes */
    .info-box {
        background: 48913A;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CACHING FUNCTIONS
# =============================================================================
@st.cache_data(show_spinner=False)
def load_data(path_interactions: str, path_items: str):
    loader = DataLoader(path_interactions, path_items)
    return loader, loader.interactions, loader.items_df

@st.cache_resource(show_spinner=True)
def build_model(interactions, items_df, n_users, n_items, alpha, half_life_days):
    model = SemanticHybridRecommender(n_users=n_users, n_items=n_items)
    model.fit(interactions, items_df, alpha=alpha, half_life_days=half_life_days)
    return model

@st.cache_resource(show_spinner=False)
def load_bert():
    if SentenceTransformer is None:
        return None
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data(show_spinner=False)
def compute_embeddings(items_df):
    model_bert = load_bert()
    if model_bert is None:
        return None
    
    df_sorted = items_df.sort_values('i_idx').fillna('')
    texts = (df_sorted['Title'] + ". " + df_sorted['Author'].fillna('') + ". " + df_sorted['Subjects'].fillna('')).tolist()
    
    # Check cache
    hasher = hashlib.md5()
    for t in texts:
        hasher.update(t.encode('utf-8', errors='ignore'))
    cache_key = hasher.hexdigest()
    
    cache_dir = os.path.join(PROJECT_ROOT, 'data', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"emb_{cache_key}.npy")
    
    if os.path.exists(cache_path):
        return np.load(cache_path)
    
    emb = model_bert.encode(texts, show_progress_bar=True)
    np.save(cache_path, emb)
    return emb

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_user_history(interactions_df, items_df, u_idx, limit=10):
    """Récupère l'historique de lecture d'un utilisateur"""
    user_ints = interactions_df[interactions_df['u_idx'] == u_idx].sort_values('t', ascending=False)
    if len(user_ints) == 0:
        return pd.DataFrame()
    
    history = user_ints.merge(items_df[['i_idx', 'Title', 'Author']], on='i_idx', how='left')
    return history[['Title', 'Author']].head(limit)

def explain_recommendation(item_row, user_history_titles, model_type="hybrid"):
    """Génère une explication pour une recommandation"""
    title = str(item_row.get('Title', ''))
    author = str(item_row.get('Author', ''))
    subjects = str(item_row.get('Subjects', ''))
    
    reasons = []
    
    # Check if same author in history
    if author and any(author.lower() in str(h).lower() for h in user_history_titles):
        reasons.append(f"📖 Vous avez déjà lu des livres de **{author}**")
    
    # Check subjects overlap
    if 'fiction' in subjects.lower():
        reasons.append("🎭 Correspond à vos goûts en fiction")
    if 'science' in subjects.lower():
        reasons.append("🔬 Thématique scientifique que vous appréciez")
    if 'history' in subjects.lower() or 'histoire' in subjects.lower():
        reasons.append("📜 Sujet historique dans vos centres d'intérêt")
    
    if not reasons:
        reasons.append("✨ Recommandé par notre algorithme hybride (similarité sémantique + comportementale)")
    
    return " • ".join(reasons)

def create_gauge_chart(value, title, max_val=1.0):
    """Crée un graphique jauge pour les métriques"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, max_val], 'tickwidth': 1},
            'bar': {'color': "#667eea"},
            'bgcolor': "white",
            'steps': [
                {'range': [0, max_val*0.5], 'color': '#f0f0f0'},
                {'range': [max_val*0.5, max_val*0.75], 'color': '#e0e0e0'},
                {'range': [max_val*0.75, max_val], 'color': '#d0d0d0'}
            ],
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def save_feedback(user_id, item_id, liked):
    """Sauvegarde le feedback utilisateur"""
    try:
        feedback = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'item_id': item_id,
            'liked': liked
        }
        df = pd.DataFrame([feedback])
        if os.path.exists(FEEDBACK_LOG):
            df.to_csv(FEEDBACK_LOG, mode='a', header=False, index=False)
        else:
            df.to_csv(FEEDBACK_LOG, index=False)
        return True
    except:
        return False

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/book-shelf.png", width=100)
    st.title("⚙️ Configuration")
    
    st.markdown("---")
    
    # Data paths
    st.subheader("📁 Données")
    path_inter = os.path.join(DATA_DIR, 'interactions_train.csv')
    path_items = os.path.join(DATA_DIR, 'items.csv')
    
    use_enriched = st.toggle("Utiliser métadonnées enrichies (IA)", value=False)
    if use_enriched:
        path_items = os.path.join(DATA_DIR, 'items_enriched_ai_turbo.csv')
    
    st.markdown("---")
    
    # Model parameters
    st.subheader("🎛️ Paramètres du Modèle")
    alpha = st.slider("Alpha (Collab ↔ Sémantique)", 0.0, 1.0, 0.5, 0.05,
                      help="0 = 100% sémantique, 1 = 100% collaboratif")
    
    col1, col2 = st.columns(2)
    with col1:
        hl_short = st.number_input("Demi-vie court (j)", 1, 30, 1)
    with col2:
        hl_long = st.number_input("Demi-vie long (j)", 50, 500, 250)
    
    k_top = st.slider("Nombre de recommandations", 5, 30, 10)
    
    st.markdown("---")
    
    # Actions
    if st.button("🔄 Reconstruire le modèle", use_container_width=True):
        build_model.clear()
        st.rerun()
    
    if st.button("🗑️ Vider le cache", use_container_width=True):
        load_data.clear()
        build_model.clear()
        compute_embeddings.clear()
        st.success("Cache vidé !")

# =============================================================================
# LOAD DATA & MODEL
# =============================================================================
try:
    loader, interactions_df, items_df = load_data(path_inter, path_items)
except Exception as e:
    st.error(f"❌ Erreur de chargement des données : {e}")
    st.info("Vérifiez que les fichiers CSV sont présents dans le dossier `data/`")
    st.stop()

with st.spinner("🧠 Chargement du modèle Super-Ensemble..."):
    model = build_model(
        interactions=loader.get_full_data(),
        items_df=items_df,
        n_users=loader.n_users,
        n_items=loader.n_items,
        alpha=alpha,
        half_life_days=[hl_short, hl_long]
    )

# Compute embeddings for semantic search
item_emb = compute_embeddings(items_df)
if item_emb is not None:
    item_emb_norm = item_emb / (np.linalg.norm(item_emb, axis=1, keepdims=True) + 1e-12)
else:
    item_emb_norm = None

# =============================================================================
# MAIN CONTENT
# =============================================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>📚 BiblioRec — Système de Recommandation Intelligent</h1>
    <p>Propulsé par le Super-Ensemble : S-BERT + Collaboratif + Séquentiel + SVD + BM25</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab_home, tab_patron, tab_discover, tab_similar, tab_stats, tab_about = st.tabs([
    "🏠 Accueil",
    "👤 Recommander à un Usager",
    "🔍 Découverte (Cold Start)",
    "📖 Livres Similaires",
    "📊 Statistiques",
    "ℹ️ À Propos"
])

# =============================================================================
# TAB: ACCUEIL
# =============================================================================
with tab_home:
    st.header("Tableau de Bord")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{loader.n_users:,}</h3>
            <p>👥 Usagers actifs</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <h3>{loader.n_items:,}</h3>
            <p>📚 Livres au catalogue</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <h3>{len(interactions_df):,}</h3>
            <p>📖 Emprunts enregistrés</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_per_user = len(interactions_df) / loader.n_users
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <h3>{avg_per_user:.1f}</h3>
            <p>📈 Emprunts/usager (moy.)</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick search
    col_search, col_results = st.columns([1, 2])
    
    with col_search:
        st.subheader("🔎 Recherche Rapide")
        search_query = st.text_input("Titre, auteur ou sujet...", placeholder="Ex: Harry Potter")
        n_results = st.slider("Résultats max", 5, 50, 20)
    
    with col_results:
        if search_query:
            q = search_query.lower()
            mask = (
                items_df['Title'].str.lower().str.contains(q, na=False) |
                items_df['Author'].fillna('').str.lower().str.contains(q, na=False) |
                items_df['Subjects'].fillna('').str.lower().str.contains(q, na=False)
            )
            results = items_df[mask][['i', 'Title', 'Author', 'Subjects']].head(n_results)
            
            if len(results) > 0:
                st.dataframe(results, use_container_width=True, hide_index=True)
            else:
                st.info("Aucun résultat trouvé.")
        else:
            st.subheader("📚 Derniers ajouts au catalogue")
            st.dataframe(
                items_df[['i', 'Title', 'Author']].tail(10).iloc[::-1],
                use_container_width=True,
                hide_index=True
            )

# =============================================================================
# TAB: RECOMMANDER À UN USAGER
# =============================================================================
with tab_patron:
    st.header("👤 Recommandations Personnalisées")
    
    col_user, col_options = st.columns([1, 1])
    
    with col_user:
        st.subheader("Sélectionner un usager")
        
        # User selection with search
        user_search = st.text_input("🔍 Rechercher par ID", "")
        
        if user_search:
            filtered_users = [u for u in loader.u_unique if str(user_search) in str(u)][:100]
        else:
            filtered_users = loader.u_unique[:100]
        
        selected_user = st.selectbox(
            "ID Usager",
            options=filtered_users,
            format_func=lambda x: f"Usager #{x}"
        )
        u_idx = loader.u_map[selected_user]
    
    with col_options:
        st.subheader("Options")
        exclude_read = st.checkbox("📕 Exclure les livres déjà empruntés", value=True)
        max_per_author = st.number_input("📚 Max par auteur (0 = illimité)", 0, 5, 2)
        show_explanations = st.checkbox("💡 Afficher les explications", value=True)
    
    st.markdown("---")
    
    # User history
    col_history, col_recs = st.columns([1, 2])
    
    with col_history:
        st.subheader("📖 Historique de l'usager")
        history = get_user_history(interactions_df, items_df, u_idx, limit=8)
        
        if len(history) > 0:
            for _, row in history.iterrows():
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 0.5rem 1rem; border-radius: 8px; margin: 0.3rem 0; border-left: 3px solid #667eea;">
                    <strong>{row['Title'][:40]}{'...' if len(str(row['Title'])) > 40 else ''}</strong><br>
                    <small style="color: #666;">{row['Author']}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Aucun historique disponible.")
    
    with col_recs:
        st.subheader("✨ Recommandations")
        
        if st.button("🎯 Générer les recommandations", type="primary", use_container_width=True):
            with st.spinner("Calcul en cours..."):
                # Get predictions
                preds = model.predict(k=k_top * 2, batch_size=1000)
                indices = preds[u_idx].tolist()
                
                # Build recommendations dataframe
                recs = items_df.set_index('i_idx').loc[indices].copy()
                
                # Apply filters
                if exclude_read:
                    read_items = interactions_df[interactions_df['u_idx'] == u_idx]['i_idx'].unique()
                    recs = recs[~recs.index.isin(read_items)]
                
                if max_per_author > 0 and 'Author' in recs.columns:
                    recs = recs.groupby('Author', group_keys=False).head(max_per_author)
                
                recs = recs.head(k_top)
                
                # Get user history titles for explanations
                history_titles = history['Title'].tolist() if len(history) > 0 else []
                
                # Display recommendations
                for rank, (idx, row) in enumerate(recs.iterrows(), 1):
                    with st.container():
                        cols = st.columns([0.5, 4, 1])
                        
                        with cols[0]:
                            st.markdown(f"### #{rank}")
                        
                        with cols[1]:
                            st.markdown(f"**{row['Title']}**")
                            st.markdown(f"*{row.get('Author', 'Auteur inconnu')}*")
                            
                            if show_explanations:
                                reason = explain_recommendation(row, history_titles)
                                st.markdown(f"""
                                <div class="book-reason">{reason}</div>
                                """, unsafe_allow_html=True)
                        
                        with cols[2]:
                            if st.button("👍", key=f"like_{idx}"):
                                save_feedback(selected_user, idx, True)
                                st.success("Merci !")
                        
                        st.markdown("---")
                
                # Download button
                csv_buf = io.StringIO()
                recs.reset_index()[['i', 'Title', 'Author']].to_csv(csv_buf, index=False)
                st.download_button(
                    "📥 Télécharger la liste (CSV)",
                    data=csv_buf.getvalue(),
                    file_name=f"recommandations_usager_{selected_user}.csv",
                    mime="text/csv"
                )

# =============================================================================
# TAB: DÉCOUVERTE (COLD START)
# =============================================================================
with tab_discover:
    st.header("🔍 Découverte — Nouvel Usager")
    
    st.markdown("""
    <div class="info-box">
        <strong>💡 Mode Cold Start</strong><br>
        Pour les nouveaux usagers sans historique, décrivez leurs goûts en quelques mots.
        Notre modèle S-BERT trouvera les livres les plus pertinents.
    </div>
    """, unsafe_allow_html=True)
    
    query = st.text_area(
        "Décrivez les intérêts de l'usager",
        placeholder="Ex: Romans policiers nordiques, enquêtes psychologiques, ambiance sombre...",
        height=100
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔮 Trouver des livres", type="primary", use_container_width=True):
            if item_emb_norm is None:
                st.error("Le modèle sémantique n'est pas disponible.")
            elif not query.strip():
                st.warning("Veuillez entrer une description.")
            else:
                with st.spinner("Analyse sémantique en cours..."):
                    bert = load_bert()
                    q_vec = bert.encode([query])
                    q_vec = q_vec / (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-12)
                    
                    sims = (q_vec @ item_emb_norm.T).ravel()
                    top_idx = np.argsort(sims)[-k_top:][::-1]
                    
                    st.session_state['cold_results'] = items_df.set_index('i_idx').loc[top_idx]
    
    with col2:
        if 'cold_results' in st.session_state:
            results = st.session_state['cold_results']
            
            for _, row in results.iterrows():
                st.markdown(f"""
                <div class="book-card">
                    <div class="book-title">{row['Title']}</div>
                    <div class="book-author">{row.get('Author', '')}</div>
                    <div style="font-size: 0.8rem; color: #888; margin-top: 0.5rem;">
                        {str(row.get('Subjects', ''))[:100]}...
                    </div>
                </div>
                """, unsafe_allow_html=True)

# =============================================================================
# TAB: LIVRES SIMILAIRES
# =============================================================================
with tab_similar:
    st.header("📖 Trouver des Livres Similaires")
    
    search_title = st.text_input("🔍 Rechercher un livre par titre", "")
    
    if search_title:
        matches = items_df[items_df['Title'].str.contains(search_title, case=False, na=False)].head(20)
        
        if len(matches) > 0:
            selected_book = st.selectbox(
                "Sélectionner le livre",
                options=matches['Title'].tolist(),
                format_func=lambda x: x[:60] + "..." if len(x) > 60 else x
            )
            
            if st.button("🔗 Trouver des livres similaires", type="primary"):
                book_row = matches[matches['Title'] == selected_book].iloc[0]
                i_idx = book_row['i_idx']
                
                # Use semantic similarity
                if item_emb_norm is not None:
                    book_vec = item_emb_norm[i_idx:i_idx+1]
                    sims = (book_vec @ item_emb_norm.T).ravel()
                    sims[i_idx] = -1  # Exclude self
                    
                    top_idx = np.argsort(sims)[-k_top:][::-1]
                    similar = items_df.set_index('i_idx').loc[top_idx]
                    
                    st.subheader(f"Livres similaires à « {selected_book[:50]}... »")
                    
                    cols = st.columns(2)
                    for i, (_, row) in enumerate(similar.iterrows()):
                        with cols[i % 2]:
                            st.markdown(f"""
                            <div class="book-card">
                                <div class="book-title">{row['Title']}</div>
                                <div class="book-author">{row.get('Author', '')}</div>
                            </div>
                            """, unsafe_allow_html=True)
        else:
            st.info("Aucun livre trouvé avec ce titre.")

# =============================================================================
# TAB: STATISTIQUES
# =============================================================================
with tab_stats:
    st.header("📊 Statistiques de la Bibliothèque")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📚 Top 15 Auteurs les plus empruntés")
        
        merged = interactions_df.merge(items_df[['i_idx', 'Author']], on='i_idx', how='left')
        top_authors = merged['Author'].fillna('Inconnu').value_counts().head(15)
        
        fig = px.bar(
            x=top_authors.values,
            y=top_authors.index,
            orientation='h',
            labels={'x': 'Nombre d\'emprunts', 'y': 'Auteur'},
            color=top_authors.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(showlegend=False, height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Distribution des emprunts par usager")
        
        user_counts = interactions_df.groupby('u_idx').size()
        
        fig = px.histogram(
            user_counts,
            nbins=50,
            labels={'value': 'Nombre d\'emprunts', 'count': 'Nombre d\'usagers'},
            color_discrete_sequence=['#667eea']
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Sparsity gauge
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sparsity = 1 - (len(interactions_df) / (loader.n_users * loader.n_items))
        fig = create_gauge_chart(sparsity * 100, "Sparsité (%)", 100)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        coverage = len(interactions_df['i_idx'].unique()) / loader.n_items
        fig = create_gauge_chart(coverage * 100, "Couverture Catalogue (%)", 100)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        active_users = len(interactions_df['u_idx'].unique()) / loader.n_users
        fig = create_gauge_chart(active_users * 100, "Usagers Actifs (%)", 100)
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB: À PROPOS
# =============================================================================
with tab_about:
    st.header("ℹ️ À Propos du Système")
    
    st.markdown("""
    ### 🧠 Architecture Super-Ensemble
    
    Ce système de recommandation combine **5 signaux complémentaires** pour des recommandations précises :
    
    | Signal | Technologie | Description |
    |--------|-------------|-------------|
    | **Sémantique** | S-BERT | Comprend le sens des titres et sujets |
    | **Collaboratif** | TF-IDF + Time-Decay | Apprend des comportements similaires |
    | **Séquentiel** | Co-visitation | Prédit le prochain livre (séries) |
    | **Latent** | SVD | Capture les patterns cachés |
    | **Lexical** | BM25 | Correspondance exacte de mots-clés |
    
    ---
    
    ### 🏆 Performance
    
    - **Score MAP@10** : 0.21181
    - **Classement Kaggle** : 1ère place
    
    ---
    
    ### 👥 Équipe
    
    **Master TIDE — Université Paris 1 Panthéon-Sorbonne**
    
    - **Sacha Jocic** — Architecture & Tuning
    - **Léa Jouffrey** — Data Science & UX
    - **Saloua Dekhissi** — Sémantique & Documentation
    
    ---
    
    ### 📚 Utilisation pour Bibliothécaires
    
    1. **Recommander à un usager** : Sélectionnez un usager existant pour voir ses recommandations personnalisées
    2. **Découverte** : Pour les nouveaux usagers, décrivez leurs goûts en texte libre
    3. **Livres similaires** : Trouvez des alternatives à un livre spécifique
    4. **Statistiques** : Analysez les tendances de votre collection
    """)
    
    st.markdown("---")
    st.caption("© 2025 BiblioRec — Projet Académique Master TIDE")
