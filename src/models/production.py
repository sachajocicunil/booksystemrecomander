import os
import json
import hashlib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Install sentence-transformers: pip install sentence-transformers")

from .base import BaseRecommender


class SemanticHybridRecommender(BaseRecommender):
    """
    MEILLEUR MODÈLE (Production) - Version "Decoupled Re-buy".

    Correction Critique :
    L'Ensemble pondéré diluait le signal de "Re-buy" des vieux items (car le modèle court terme les oublie).
    Cette version applique le boost de Re-buy UNIQUEMENT sur l'historique le plus long,
    tout en utilisant l'ensemble court/long pour la similarité.
    """

    def __init__(self, n_users, n_items):
        super().__init__(n_users, n_items)
        # Compatibilité
        self.train_matrix_tfidf = None
        self.item_similarity = None

        # Ensemble
        self.ensemble_models = []
        self.pop_scores = None
        self.long_term_user_matrix = None  # Stocke l'historique le plus complet

    def fit(self, df_interactions, df_items, alpha=0.5, half_life_days=[1, 250], ensemble_weights=[0.5, 0.5]):
        """
        Entraîne le modèle hybride décorrélé pour le re-buy.

        Paramètres
        ----------
        df_interactions : pd.DataFrame
            Interactions avec colonnes `u_idx`, `i_idx`, `t`. Peut contenir des duplicats/poids, ici pondérés par time-decay.
        df_items : pd.DataFrame
            Métadonnées items alignées (incluant `i_idx`) avec colonnes textuelles `Title`, `Author`, `Subjects`.
        alpha : float in [0,1]
            Poids de la composante collaborative vs sémantique (alpha*collab + (1-alpha)*content).
        half_life_days : int | list[int]
            Une ou plusieurs demi‑vies (en jours) pour construire des sous‑modèles pondérés par récence.
        ensemble_weights : list[float] | None
            Poids de chaque sous‑modèle dans l’agrégation. Par défaut: uniforme.
        """
        if isinstance(half_life_days, (int, float)):
            half_life_days = [half_life_days]

        if ensemble_weights is None:
            ensemble_weights = [1.0 / len(half_life_days)] * len(half_life_days)

        if len(ensemble_weights) != len(half_life_days):
            print(f"⚠️ Warning: Weights len != Half-lives len. Using equal weights.")
            ensemble_weights = [1.0 / len(half_life_days)] * len(half_life_days)

        print(f"Fitting SemanticHybrid Decoupled | Alpha={alpha}, HL={half_life_days}, Weights={ensemble_weights}...")

        # --- 1. S-BERT (Commun) avec cache disque des embeddings ---
        print("Loading S-BERT & preparing item embeddings (with disk cache)...")
        model_name = 'all-MiniLM-L6-v2'
        model_bert = SentenceTransformer(model_name)
        df_items_sorted = df_items.sort_values('i_idx').copy()
        # Garantir la présence des colonnes texte
        for col in ['Title', 'Author', 'Subjects']:
            if col not in df_items_sorted.columns:
                df_items_sorted[col] = ''
        df_items_sorted = df_items_sorted.fillna('')

        # Construire les textes et un hash stable dépendant de l'ordre i_idx
        hasher = hashlib.md5()
        texts = []
        for _, row in df_items_sorted.iterrows():
            t = f"{row.get('Title','')}. {row.get('Author','')}. {row.get('Subjects','')}"
            texts.append(t)
            hasher.update(t.encode('utf-8', errors='ignore'))
        hash_key = hasher.hexdigest()

        # Dossier de cache: <repo>/data/cache
        cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'cache'))
        os.makedirs(cache_dir, exist_ok=True)
        base = f"embeddings_{model_name}_{hash_key}"
        base = base.replace('/', '_')
        path_npy = os.path.join(cache_dir, base + '.npy')
        path_meta = os.path.join(cache_dir, base + '.json')

        item_embeddings = None
        # Essayer de charger du cache
        if os.path.exists(path_npy) and os.path.exists(path_meta):
            try:
                with open(path_meta, 'r') as f:
                    meta = json.load(f)
                if meta.get('n_items') == len(texts) and meta.get('model') == model_name:
                    item_embeddings = np.load(path_npy, mmap_mode='r')
                    print(f"Loaded embeddings from cache: {path_npy}")
            except Exception as e:
                print(f"Cache load failed, will recompute. Reason: {e}")
                item_embeddings = None

        # Si pas de cache, encoder et sauver
        if item_embeddings is None:
            item_embeddings = model_bert.encode(texts, show_progress_bar=True)
            try:
                np.save(path_npy, item_embeddings)
                with open(path_meta, 'w') as f:
                    json.dump({'model': model_name, 'n_items': int(len(texts)), 'hash': hash_key}, f)
                print(f"Saved embeddings to cache: {path_npy}")
            except Exception as e:
                print(f"Warning: failed to save embeddings cache: {e}")

        sim_content = cosine_similarity(item_embeddings)
        
        # --- 1.5 BM25 / TF-IDF Keyword Similarity ---
        print("Computing BM25/TF-IDF Keyword Similarity...")
        tfidf_bm25 = TfidfVectorizer(stop_words='english', min_df=2)
        bm25_matrix = tfidf_bm25.fit_transform(texts)
        self.sim_bm25 = cosine_similarity(bm25_matrix, dense_output=True)

        # --- 2. ENSEMBLE LOOP ---
        self.ensemble_models = []
        max_hl = -1
        min_hl = float('inf')
        short_term_user_matrix = None

        for idx, hl in enumerate(half_life_days):
            print(f"\n--- Building Sub-Model {idx + 1} (Half-life={hl}d) ---")

            # A. Time Decay
            df = df_interactions.copy()
            df['last_user_ts'] = df.groupby('u_idx')['t'].transform('max')
            df['days_diff'] = (df['last_user_ts'] - df['t']) / (24 * 3600)
            decay_rate = np.log(2) / hl
            df['weight'] = np.exp(-decay_rate * df['days_diff'])

            row = df['u_idx'].values
            col = df['i_idx'].values
            data = df['weight'].values

            matrix_sparse = sparse.csr_matrix((data, (row, col)), shape=(self.n_users, self.n_items))

            # TF-IDF
            tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
            user_profile = tfidf.fit_transform(matrix_sparse)

            # Collab Sim
            sim_collaborative = cosine_similarity(user_profile.T, dense_output=False)

            # B. Fusion (SANS le boost diagonal ici)
            sim_final = (sim_collaborative * alpha) + (sim_content * (1 - alpha))

            # On stocke le modèle
            self.ensemble_models.append({
                'user_matrix': user_profile,
                'item_matrix': sim_final,
                'weight': ensemble_weights[idx]
            })

            # On repère l'historique le plus long (le plus grand HL)
            if hl > max_hl:
                max_hl = hl
                self.long_term_user_matrix = user_profile

            # Compatibilité
            self.train_matrix_tfidf = user_profile
            self.item_similarity = sim_final

        # --- 3. POPULARITY ---
        print(f"\nComputing Global Popularity Scores (Based on HL={max_hl}d)...")
        # Utiliser le court terme pour la popularité (Trending) au lieu du long terme
        item_popularity = np.array(self.long_term_user_matrix.sum(axis=0)).flatten()
        self.pop_scores = item_popularity / item_popularity.max() if item_popularity.max() > 0 else item_popularity

        print("Ensemble Model Fitted Successfully.")

    def predict(self, k=10, batch_size=1000, re_buy_factor=0.5, pop_factor=0.2, bm25_weight=0.15):
        predictions = []

        for start_idx in range(0, self.n_users, batch_size):
            end_idx = min(start_idx + batch_size, self.n_users)

            # 1. Calcul des scores de similarité (Exploration)
            final_batch_scores = None

            for model in self.ensemble_models:
                user_batch = model['user_matrix'][start_idx:end_idx]
                sim_matrix = model['item_matrix']
                weight = model['weight']

                # Score = User * Sim (Attention: Sim n'a PAS de boost diagonal ici)
                scores = user_batch.dot(sim_matrix)

                if sparse.issparse(scores): scores = scores.toarray()
                scores = np.asarray(scores)

                if final_batch_scores is None:
                    final_batch_scores = scores * weight
                else:
                    final_batch_scores += scores * weight

            # 2. Ajout du Re-Buy Boost (Exploitation)
            # On prend l'historique LONG TERME pour ne pas oublier les vieux items
            # On ajoute re_buy_factor * Historique à la fin
            long_term_batch = self.long_term_user_matrix[start_idx:end_idx]
            if sparse.issparse(long_term_batch):
                long_term_batch = long_term_batch.toarray()

            # Le facteur re_buy_factor correspond à ton boost diagonal précédent
            final_batch_scores += (re_buy_factor * long_term_batch)

            # 3. Boost Popularité
            if self.pop_scores is not None:
                final_batch_scores += (pop_factor * self.pop_scores.reshape(1, -1))
            
            # 4. Boost BM25 (Keyword Matching)
            if self.sim_bm25 is not None:
                # On utilise l'historique long terme pour matcher les keywords
                scores_bm25 = long_term_batch.dot(self.sim_bm25)
                if sparse.issparse(scores_bm25): scores_bm25 = scores_bm25.toarray()
                final_batch_scores += (bm25_weight * scores_bm25)

            # Sélection Top K
            top_k_unsorted = np.argpartition(final_batch_scores, -k, axis=1)[:, -k:]

            batch_preds = []
            for i in range(len(final_batch_scores)):
                row_scores = final_batch_scores[i]
                idx = top_k_unsorted[i]
                sorted_idx = idx[np.argsort(row_scores[idx])[::-1]]
                batch_preds.append(sorted_idx)

            predictions.extend(batch_preds)

        return np.array(predictions)

    def get_batch_scores(self, start_idx, end_idx, re_buy_factor=0.5, pop_factor=0.2):
        """
        Retourne la matrice de scores denses pour un batch d'utilisateurs.
        Utilisé pour l'Ensembling.
        """
        # 1. Calcul des scores de similarité (Exploration)
        final_batch_scores = None

        for model in self.ensemble_models:
            user_batch = model['user_matrix'][start_idx:end_idx]
            sim_matrix = model['item_matrix']
            weight = model['weight']

            scores = user_batch.dot(sim_matrix)

            if sparse.issparse(scores): scores = scores.toarray()
            scores = np.asarray(scores)

            if final_batch_scores is None:
                final_batch_scores = scores * weight
            else:
                final_batch_scores += scores * weight

        # 2. Ajout du Re-Buy Boost
        long_term_batch = self.long_term_user_matrix[start_idx:end_idx]
        if sparse.issparse(long_term_batch):
            long_term_batch = long_term_batch.toarray()

        final_batch_scores += (re_buy_factor * long_term_batch)

        # 3. Boost Popularité
        if self.pop_scores is not None:
            final_batch_scores += (pop_factor * self.pop_scores.reshape(1, -1))
            
        return final_batch_scores