import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

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

    def fit(self, df_interactions, df_items, alpha=0.5, half_life_days=[30, 150], ensemble_weights=None):
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

        # --- 1. S-BERT (Commun) ---
        print("Loading S-BERT & Encoding Metadata...")
        model_bert = SentenceTransformer('all-MiniLM-L6-v2')
        df_items_sorted = df_items.sort_values('i_idx').fillna('')
        soup_bert = (df_items_sorted['Title'] + ". " + df_items_sorted['Author'] + ". " + df_items_sorted[
            'Subjects']).tolist()
        item_embeddings = model_bert.encode(soup_bert, show_progress_bar=True)
        sim_content = cosine_similarity(item_embeddings)

        # --- 2. ENSEMBLE LOOP ---
        self.ensemble_models = []
        max_hl = -1

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
        print("\nComputing Global Popularity Scores...")
        item_popularity = np.array(self.long_term_user_matrix.sum(axis=0)).flatten()
        self.pop_scores = item_popularity / item_popularity.max() if item_popularity.max() > 0 else item_popularity

        print("Ensemble Model Fitted Successfully.")

    def predict(self, k=10, batch_size=1000):
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
            # On ajoute 1.5 * Historique à la fin
            long_term_batch = self.long_term_user_matrix[start_idx:end_idx]
            if sparse.issparse(long_term_batch):
                long_term_batch = long_term_batch.toarray()

            # Le facteur 1.5 correspond à ton boost diagonal précédent
            final_batch_scores += (1.5 * long_term_batch)

            # 3. Boost Popularité
            if self.pop_scores is not None:
                final_batch_scores += (0.1 * self.pop_scores.reshape(1, -1))

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