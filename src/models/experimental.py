import numpy as np
import scipy.linalg
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseRecommender

# Tentative d'import conditionnel pour S-BERT (utilisé par l'Expérience 9)
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# ==============================================================================
# EXPÉRIENCE 1 : BM25 (Probabilistic Model)
# Hypothèse : BM25 gère mieux la saturation des fréquences que TF-IDF.
# Résultat : Pas d'amélioration significative, calcul plus lourd.
# ==============================================================================
class BM25Transformer:
    """Helper class pour la transformation BM25"""

    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b

    def fit_transform(self, X):
        N = X.shape[0]
        idf = np.log(1 + (N - X.getnnz(axis=0) + 0.5) / (X.getnnz(axis=0) + 0.5))
        self.idf_diag = sparse.diags(idf)
        dl = X.sum(axis=1).A1
        avgdl = dl.mean()
        data = X.data
        rows, cols = X.nonzero()
        new_data = data * (self.k1 + 1) / (data + self.k1 * (1 - self.b + self.b * dl[rows] / avgdl))
        X_bm25 = sparse.csr_matrix((new_data, (rows, cols)), shape=X.shape)
        return X_bm25 * self.idf_diag


class BM25Recommender(BaseRecommender):
    def __init__(self, n_users, n_items):
        super().__init__(n_users, n_items)
        self.train_matrix_bm25 = None
        self.item_similarity = None

    def fit(self, df_interactions, df_items, k1=1.2, b=0.75, half_life_days=30):
        print(f"Fitting EXP: BM25 Model (k1={k1}, b={b})...")
        # Time Decay
        df = df_interactions.copy()
        df['last_user_ts'] = df.groupby('u_idx')['t'].transform('max')
        df['days_diff'] = (df['last_user_ts'] - df['t']) / (24 * 3600)
        df['weight'] = np.exp(-(np.log(2) / half_life_days) * df['days_diff'])

        row = df['u_idx'].values
        col = df['i_idx'].values
        matrix_sparse = sparse.csr_matrix((df['weight'], (row, col)), shape=(self.n_users, self.n_items))

        # Transformation BM25 au lieu de TF-IDF
        self.train_matrix_bm25 = BM25Transformer(k1, b).fit_transform(matrix_sparse)
        self.item_similarity = cosine_similarity(self.train_matrix_bm25.T, dense_output=False)

    def predict(self, k=10, batch_size=1000):
        predictions = []
        for start in range(0, self.n_users, batch_size):
            end = min(start + batch_size, self.n_users)
            scores = self.train_matrix_bm25[start:end].dot(self.item_similarity).toarray()
            top_k = np.argpartition(scores, -k, axis=1)[:, -k:]
            batch_preds = [idx[np.argsort(scores[i][idx])[::-1]] for i, idx in enumerate(top_k)]
            predictions.extend(batch_preds)
        return np.array(predictions)


# ==============================================================================
# EXPÉRIENCE 2 : EASE (Embarrassingly Shallow Autoencoders)
# Hypothèse : Apprendre les poids d'interaction directement (Matrice Gram).
# Résultat : Très lourd en RAM, tendance à sur-apprendre sur ce dataset sparse.
# ==============================================================================
class EASERecommender(BaseRecommender):
    def __init__(self, n_users, n_items):
        super().__init__(n_users, n_items)
        self.train_matrix = None
        self.item_similarity = None  # Contient la matrice B de poids EASE

    def fit(self, df_interactions, df_items, lambda_reg=500, half_life_days=30):
        print(f"Fitting EXP: EASE Model (Lambda={lambda_reg})...")
        # Time Decay + TFIDF preparation
        df = df_interactions.copy()
        df['last_user_ts'] = df.groupby('u_idx')['t'].transform('max')
        df['days_diff'] = (df['last_user_ts'] - df['t']) / (24 * 3600)
        df['weight'] = np.exp(-(np.log(2) / half_life_days) * df['days_diff'])
        matrix_sparse = sparse.csr_matrix((df['weight'], (df['u_idx'], df['i_idx'])),
                                          shape=(self.n_users, self.n_items))
        self.train_matrix = TfidfTransformer().fit_transform(matrix_sparse)

        # EASE Math: B = P / -diag(P) where P = (X^T X + lambda I)^-1
        print(" Computing Gram Matrix (Heavy RAM)...")
        G = self.train_matrix.T.dot(self.train_matrix).toarray()
        diag_indices = np.diag_indices(G.shape[0])
        G[diag_indices] += lambda_reg
        P = scipy.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diag_indices] = 0
        self.item_similarity = sparse.csr_matrix(B)

    def predict(self, k=10, batch_size=1000):
        predictions = []
        for start in range(0, self.n_users, batch_size):
            end = min(start + batch_size, self.n_users)
            # Score = X . B
            scores = self.train_matrix[start:end].dot(self.item_similarity).toarray()
            top_k = np.argpartition(scores, -k, axis=1)[:, -k:]
            batch_preds = [idx[np.argsort(scores[i][idx])[::-1]] for i, idx in enumerate(top_k)]
            predictions.extend(batch_preds)
        return np.array(predictions)


# ==============================================================================
# EXPÉRIENCE 3 : SVD (Singular Value Decomposition)
# Hypothèse : Les facteurs latents capturent mieux les "concepts" cachés.
# Résultat : Score inférieur. La sémantique (Contenu) semble plus importante que le latent pur.
# ==============================================================================
class SVDRecommender(BaseRecommender):
    def __init__(self, n_users, n_items):
        super().__init__(n_users, n_items)
        self.u_factors = None
        self.vt_factors = None
        self.sigma = None

    def fit(self, df_interactions, df_items, n_factors=20):
        print(f"Fitting EXP: SVD Model (Factors={n_factors})...")
        row = df_interactions['u_idx'].values
        col = df_interactions['i_idx'].values
        data = np.ones(len(df_interactions))
        matrix_sparse = sparse.csr_matrix((data, (row, col)), shape=(self.n_users, self.n_items)).astype(float)

        # SVDs
        u, s, vt = svds(matrix_sparse, k=n_factors)
        idx = np.argsort(s)[::-1]
        self.u_factors = u[:, idx]
        self.sigma = np.diag(s[idx])
        self.vt_factors = vt[idx, :]

    def predict(self, k=10, batch_size=1000):
        predictions = []
        right_term = self.sigma @ self.vt_factors
        for start in range(0, self.n_users, batch_size):
            end = min(start + batch_size, self.n_users)
            # Score = U . Sigma . V^T
            batch_u = self.u_factors[start:end, :]
            scores = batch_u @ right_term
            top_k = np.argpartition(scores, -k, axis=1)[:, -k:]
            batch_preds = [idx[np.argsort(scores[i][idx])[::-1]] for i, idx in enumerate(top_k)]
            predictions.extend(batch_preds)
        return np.array(predictions)


# ==============================================================================
# EXPÉRIENCE 4 : Diversification (Filtrage Auteur)
# Hypothèse : Empêcher d'avoir 10 livres du même auteur.
# Résultat : Baisse du MAP@K car les items pertinents étaient filtrés.
# ==============================================================================
class DiversifiedRecommender(BaseRecommender):
    def __init__(self, n_users, n_items):
        super().__init__(n_users, n_items)
        # On utilise une version simple du modèle time decay en interne
        self.train_matrix = None
        self.item_similarity = None
        self.item_authors = None

    def fit(self, df_interactions, df_items, half_life_days=30):
        print("Fitting EXP: Diversified Model (Max 2 books/author)...")
        # 1. Base Model Logic (Simplified)
        df = df_interactions.copy()
        df['last_user_ts'] = df.groupby('u_idx')['t'].transform('max')
        df['weight'] = np.exp(-(np.log(2) / half_life_days) * ((df['last_user_ts'] - df['t']) / (24 * 3600)))
        matrix = sparse.csr_matrix((df['weight'], (df['u_idx'], df['i_idx'])), shape=(self.n_users, self.n_items))
        self.train_matrix = TfidfTransformer().fit_transform(matrix)
        self.item_similarity = cosine_similarity(self.train_matrix.T, dense_output=False)

        # 2. Store Authors
        self.item_authors = df_items.sort_values('i_idx')['Author'].fillna('Unknown').values

    def predict(self, k=10, batch_size=1000):
        predictions = []
        # On demande plus de candidats (3*k) pour avoir de la marge après filtrage
        candidates_k = k * 3

        for start in range(0, self.n_users, batch_size):
            end = min(start + batch_size, self.n_users)
            scores = self.train_matrix[start:end].dot(self.item_similarity).toarray()

            top_unsorted = np.argpartition(scores, -candidates_k, axis=1)[:, -candidates_k:]

            batch_preds = []
            for i in range(len(scores)):
                row_scores = scores[i]
                idx = top_unsorted[i]
                sorted_candidates = idx[np.argsort(row_scores[idx])[::-1]]

                # --- Logique de Diversification ---
                refined_preds = []
                author_counts = {}

                for item_id in sorted_candidates:
                    auth = self.item_authors[item_id]
                    if auth != 'Unknown' and author_counts.get(auth, 0) >= 2:
                        continue  # Skip (trop de livres de cet auteur)

                    refined_preds.append(item_id)
                    author_counts[auth] = author_counts.get(auth, 0) + 1
                    if len(refined_preds) == k: break

                # Remplissage si filtrage trop agressif
                if len(refined_preds) < k:
                    remaining = [x for x in sorted_candidates if x not in refined_preds]
                    refined_preds.extend(remaining[:k - len(refined_preds)])

                batch_preds.append(refined_preds)
            predictions.extend(batch_preds)
        return np.array(predictions)


# ==============================================================================
# EXPÉRIENCE 5 : Ensemble Short-Term / Long-Term
# Hypothèse : Combiner un modèle "récent" (7j) et un "historique" (90j).
# Résultat : Complexité accrue sans gain significatif par rapport au modèle unique (30j).
# ==============================================================================
class EnsembleHybridRecommender(BaseRecommender):
    def __init__(self, n_users, n_items):
        super().__init__(n_users, n_items)
        self.models = []
        self.pop_scores = None

    def fit(self, df_interactions, df_items, alpha=0.8, half_lives=[7, 90], lambda_reg=500):
        print(f"Fitting EXP: Ensemble Model with Half-lives: {half_lives}...")

        # Poids de l'ensemble (ex: 60% Short, 40% Long)
        ensemble_weights = [0.6, 0.4] if len(half_lives) == 2 else [1.0 / len(half_lives)] * len(half_lives)

        # 1. Content-Based (Commun)
        from sklearn.feature_extraction.text import TfidfVectorizer
        df_items_sorted = df_items.sort_values('i_idx').fillna('')
        df_items_sorted['metadata_soup'] = (
                df_items_sorted['Title'] + " " +
                (df_items_sorted['Author'] + " ") * 3 +
                (df_items_sorted['Subjects'] + " ") * 2 +
                df_items_sorted['Publisher']
        )
        vec = TfidfVectorizer(stop_words='english', min_df=2, ngram_range=(1, 2))
        item_features = vec.fit_transform(df_items_sorted['metadata_soup'])
        sim_content = cosine_similarity(item_features, dense_output=False)

        # 2. Entraînement des sous-modèles
        for idx, hl in enumerate(half_lives):
            print(f"   -> Sub-model {hl} days...")
            # Decay
            df = df_interactions.copy()
            df['last_user_ts'] = df.groupby('u_idx')['t'].transform('max')
            df['weight'] = np.exp(-(np.log(2) / hl) * ((df['last_user_ts'] - df['t']) / (24 * 3600)))

            matrix = sparse.csr_matrix((df['weight'], (df['u_idx'], df['i_idx'])), shape=(self.n_users, self.n_items))
            user_profile = TfidfTransformer().fit_transform(matrix)

            # EASE partiel pour la similarité
            G = user_profile.T.dot(user_profile).toarray()
            di = np.diag_indices(G.shape[0])
            G[di] += lambda_reg
            P = scipy.linalg.inv(G)
            B = P / (-np.diag(P))
            B[di] = 0
            sim_collab = sparse.csr_matrix(B)

            # Normalisation et Fusion
            if np.abs(sim_collab).max() > 0:
                sim_collab = sim_collab / np.abs(sim_collab).max()

            sim_final = (sim_collab * alpha) + (sim_content * (1 - alpha))

            # Re-buy boost
            rng = np.arange(self.n_items)
            if sparse.issparse(sim_final):
                sim_final.setdiag(sim_final.diagonal() + 1.5)
            else:
                sim_final[rng, rng] += 1.5

            self.models.append({
                'user_matrix': user_profile,
                'item_matrix': sim_final,
                'weight': ensemble_weights[idx]
            })

        # Popularité (basée sur le dernier modèle pour simplifier)
        pop = np.array(self.models[-1]['user_matrix'].sum(axis=0)).flatten()
        self.pop_scores = pop / pop.max()

    def predict(self, k=10, batch_size=1000):
        predictions = []
        for start in range(0, self.n_users, batch_size):
            end = min(start + batch_size, self.n_users)

            final_batch_scores = None

            # Somme pondérée des scores de chaque sous-modèle
            for m in self.models:
                user_batch = m['user_matrix'][start:end]
                scores = user_batch.dot(m['item_matrix'])
                if sparse.issparse(scores): scores = scores.toarray()

                if final_batch_scores is None:
                    final_batch_scores = scores * m['weight']
                else:
                    final_batch_scores += scores * m['weight']

            # Bonus Popularité
            final_batch_scores += (0.1 * self.pop_scores)

            top_k = np.argpartition(final_batch_scores, -k, axis=1)[:, -k:]
            batch_preds = [idx[np.argsort(final_batch_scores[i][idx])[::-1]] for i, idx in enumerate(top_k)]
            predictions.extend(batch_preds)
        return np.array(predictions)


# ==============================================================================
# EXPÉRIENCE 6 : Filtrage Strict "Déjà Vu" (-Infini)
# Hypothèse : Interdire formellement de recommander un item déjà consommé.
# Résultat : Baisse du score, car le "re-buy" (relecture) est un comportement valide ici.
# ==============================================================================
class HistoryFilterRecommender(BaseRecommender):
    def __init__(self, n_users, n_items):
        super().__init__(n_users, n_items)
        self.train_matrix = None
        self.item_similarity = None

    def fit(self, df_interactions, df_items, half_life_days=30):
        print("Fitting EXP: History Filter Model (No Re-buy allowed)...")
        # Standard Time Decay fit (simplifié pour l'exemple)
        df = df_interactions.copy()
        df['last_user_ts'] = df.groupby('u_idx')['t'].transform('max')
        df['weight'] = np.exp(-(np.log(2) / half_life_days) * ((df['last_user_ts'] - df['t']) / (24 * 3600)))
        matrix = sparse.csr_matrix((df['weight'], (df['u_idx'], df['i_idx'])), shape=(self.n_users, self.n_items))

        self.train_matrix = TfidfTransformer().fit_transform(matrix)
        self.item_similarity = cosine_similarity(self.train_matrix.T, dense_output=False)

    def predict(self, k=10, batch_size=1000):
        predictions = []
        for start in range(0, self.n_users, batch_size):
            end = min(start + batch_size, self.n_users)
            user_batch = self.train_matrix[start:end]
            scores = user_batch.dot(self.item_similarity).toarray()

            # --- FILTRAGE AGRESSIF ---
            # On trouve les indices où l'utilisateur a déjà interagi
            rows_seen, cols_seen = user_batch.nonzero()
            # On force le score à -Infini
            scores[rows_seen, cols_seen] = -np.inf
            # -------------------------

            top_k = np.argpartition(scores, -k, axis=1)[:, -k:]
            batch_preds = [idx[np.argsort(scores[i][idx])[::-1]] for i, idx in enumerate(top_k)]
            predictions.extend(batch_preds)
        return np.array(predictions)


# ==============================================================================
# EXPÉRIENCE 7 : Filtrage Items Rares (< 3 interactions)
# Hypothèse : Supprimer le bruit des items vus 1 ou 2 fois avant l'apprentissage.
# Résultat : Perte d'information (Cold start items) nuisible au score global.
# ==============================================================================
class LowInteractionRecommender(BaseRecommender):
    def __init__(self, n_users, n_items):
        super().__init__(n_users, n_items)
        self.train_matrix = None
        self.item_similarity = None

    def fit(self, df_interactions, df_items, min_interactions=3):
        print(f"Fitting EXP: Low Interaction Filter (Min {min_interactions} views)...")
        df = df_interactions.copy()

        # --- FILTRAGE AVANT FIT ---
        item_counts = df['i_idx'].value_counts()
        valid_items = item_counts[item_counts >= min_interactions].index
        df_filtered = df[df['i_idx'].isin(valid_items)].copy()
        print(f"   -> Dropped {len(df) - len(df_filtered)} interactions on rare items.")

        # Standard logic ensuite
        df_filtered['last_user_ts'] = df_filtered.groupby('u_idx')['t'].transform('max')
        df_filtered['weight'] = np.exp(
            -(np.log(2) / 30) * ((df_filtered['last_user_ts'] - df_filtered['t']) / (24 * 3600)))

        matrix = sparse.csr_matrix((df_filtered['weight'], (df_filtered['u_idx'], df_filtered['i_idx'])),
                                   shape=(self.n_users, self.n_items))
        self.train_matrix = TfidfTransformer().fit_transform(matrix)
        self.item_similarity = cosine_similarity(self.train_matrix.T, dense_output=False)

    def predict(self, k=10, batch_size=1000):
        # Standard predict
        predictions = []
        for start in range(0, self.n_users, batch_size):
            end = min(start + batch_size, self.n_users)
            scores = self.train_matrix[start:end].dot(self.item_similarity).toarray()
            top_k = np.argpartition(scores, -k, axis=1)[:, -k:]
            batch_preds = [idx[np.argsort(scores[i][idx])[::-1]] for i, idx in enumerate(top_k)]
            predictions.extend(batch_preds)
        return np.array(predictions)


# ==============================================================================
# EXPÉRIENCE 8 : Coupled Semantic Re-buy (Archived)
# Hypothèse : Appliquer le re-buy dans chaque sous-modèle de l'ensemble S-BERT.
# Résultat : ÉCHEC (Dilution du signal de rappel par le modèle court terme).
# ==============================================================================
class CoupledSemanticRecommender(BaseRecommender):
    def __init__(self, n_users, n_items):
        super().__init__(n_users, n_items)
        self.ensemble_models = []
        self.pop_scores = None

    def fit(self, df_interactions, df_items, alpha=0.5, half_life_days=[30, 150], ensemble_weights=None):
        if SentenceTransformer is None:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")

        if ensemble_weights is None: ensemble_weights = [1.0 / len(half_life_days)] * len(half_life_days)
        print("Fitting EXP: Coupled Semantic Ensemble (Flawed Logic)...")

        model_bert = SentenceTransformer('all-MiniLM-L6-v2')
        df_items_sorted = df_items.sort_values('i_idx').fillna('')
        soup_bert = (df_items_sorted['Title'] + ". " + df_items_sorted['Author'] + ". " + df_items_sorted[
            'Subjects']).tolist()
        item_embeddings = model_bert.encode(soup_bert, show_progress_bar=False)
        sim_content = cosine_similarity(item_embeddings)

        self.ensemble_models = []
        for idx, hl in enumerate(half_life_days):
            df = df_interactions.copy()
            df['last_user_ts'] = df.groupby('u_idx')['t'].transform('max')
            df['weight'] = np.exp(-(np.log(2) / hl) * ((df['last_user_ts'] - df['t']) / (24 * 3600)))
            matrix = sparse.csr_matrix((df['weight'], (df['u_idx'], df['i_idx'])), shape=(self.n_users, self.n_items))
            user_profile = TfidfTransformer().fit_transform(matrix)
            sim_collab = cosine_similarity(user_profile.T, dense_output=False)

            sim_final = (sim_collab * alpha) + (sim_content * (1 - alpha))
            # LE DÉFAUT EST ICI : Application du boost DANS la boucle
            rng = np.arange(self.n_items)
            if sparse.issparse(sim_final):
                sim_final.setdiag(sim_final.diagonal() + 1.5)
            else:
                sim_final[rng, rng] += 1.5

            self.ensemble_models.append(
                {'user_matrix': user_profile, 'item_matrix': sim_final, 'weight': ensemble_weights[idx]})

        pop = np.array(self.ensemble_models[-1]['user_matrix'].sum(axis=0)).flatten()
        self.pop_scores = pop / pop.max()

    def predict(self, k=10, batch_size=1000):
        predictions = []
        for start in range(0, self.n_users, batch_size):
            end = min(start + batch_size, self.n_users)
            final_batch_scores = None
            for m in self.ensemble_models:
                scores = m['user_matrix'][start:end].dot(m['item_matrix'])
                if sparse.issparse(scores): scores = scores.toarray()
                scores = np.asarray(scores)
                if final_batch_scores is None:
                    final_batch_scores = scores * m['weight']
                else:
                    final_batch_scores += scores * m['weight']

            final_batch_scores += (0.1 * self.pop_scores)
            top_k = np.argpartition(final_batch_scores, -k, axis=1)[:, -k:]
            batch_preds = [idx[np.argsort(final_batch_scores[i][idx])[::-1]] for i, idx in enumerate(top_k)]
            predictions.extend(batch_preds)
        return np.array(predictions)


# ==============================================================================
# EXPÉRIENCE 9 : Semantic Hybrid (Enrichissement ChatGPT)
# Description : Version "Decoupled Re-buy" avec métadonnées enrichies (Description + Auteur Nettoyé)
# Note : C'est le "Meilleur Modèle" (Production) archivé ici.
# ==============================================================================
class SemanticHybridRecommenderChatGPT(BaseRecommender):
    """
    MEILLEUR MODÈLE (Production) - Version "Decoupled Re-buy"
    MISE A JOUR : Support des métadonnées enrichies (Description + Auteur Nettoyé)
    """

    def __init__(self, n_users, n_items):
        super().__init__(n_users, n_items)
        self.train_matrix_tfidf = None
        self.item_similarity = None
        self.ensemble_models = []
        self.pop_scores = None
        self.long_term_user_matrix = None

    def fit(self, df_interactions, df_items, alpha=0.5, half_life_days=[30, 150], ensemble_weights=None):
        if SentenceTransformer is None:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")

        # ... (Gestion des poids inchangée) ...
        if isinstance(half_life_days, (int, float)):
            half_life_days = [half_life_days]
        if ensemble_weights is None:
            ensemble_weights = [1.0 / len(half_life_days)] * len(half_life_days)
        if len(ensemble_weights) != len(half_life_days):
            ensemble_weights = [1.0 / len(half_life_days)] * len(half_life_days)

        print(f"Fitting SemanticHybrid Decoupled | Alpha={alpha}, HL={half_life_days}...")

        # --- 1. S-BERT (PARTIE MODIFIÉE) ---
        print("Loading S-BERT & Encoding ENRICHED Metadata...")
        model_bert = SentenceTransformer('all-MiniLM-L6-v2')

        # On trie pour être sûr de l'ordre
        df_items_sorted = df_items.sort_values('i_idx').copy()

        # 1. Gestion des colonnes manquantes (Sécurité si le CSV n'est pas parfait)
        # On remplit les NaNs par du vide
        for col in ['Title', 'Author', 'Subjects', 'description', 'clean_author', 'category']:
            if col in df_items_sorted.columns:
                df_items_sorted[col] = df_items_sorted[col].fillna('')
            else:
                df_items_sorted[col] = ''  # Si la colonne n'existe pas, on met du vide

        # 2. Construction de la "Super Soup"
        # On privilégie 'clean_author' s'il existe, sinon 'Author'
        # On ajoute la description qui est le game-changer

        soup_bert = (
                "Titre: " + df_items_sorted['Title'] + ". " +
                "Auteur: " + df_items_sorted['clean_author'] + ". " +
                "Genre: " + df_items_sorted['category'] + ". " +
                "Sujets: " + df_items_sorted['Subjects'] + ". " +
                "Résumé: " + df_items_sorted['description']
        ).tolist()

        # Encodage (C'est là que la magie opère)
        item_embeddings = model_bert.encode(soup_bert, show_progress_bar=True)
        sim_content = cosine_similarity(item_embeddings)

        # --- 2. ENSEMBLE LOOP (Reste inchangé) ---
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

            # TF-IDF & Collab
            tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
            user_profile = tfidf.fit_transform(matrix_sparse)
            sim_collaborative = cosine_similarity(user_profile.T, dense_output=False)

            # B. Fusion
            sim_final = (sim_collaborative * alpha) + (sim_content * (1 - alpha))

            self.ensemble_models.append({
                'user_matrix': user_profile,
                'item_matrix': sim_final,
                'weight': ensemble_weights[idx]
            })

            if hl > max_hl:
                max_hl = hl
                self.long_term_user_matrix = user_profile

        # --- 3. POPULARITY ---
        print("\nComputing Global Popularity Scores...")
        item_popularity = np.array(self.long_term_user_matrix.sum(axis=0)).flatten()
        self.pop_scores = item_popularity / item_popularity.max() if item_popularity.max() > 0 else item_popularity

        print("Ensemble Model Fitted Successfully.")

    def predict(self, k=10, batch_size=1000):
        predictions = []
        for start_idx in range(0, self.n_users, batch_size):
            end_idx = min(start_idx + batch_size, self.n_users)
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
            long_term_batch = self.long_term_user_matrix[start_idx:end_idx]
            if sparse.issparse(long_term_batch):
                long_term_batch = long_term_batch.toarray()
            final_batch_scores += (1.5 * long_term_batch)
            if self.pop_scores is not None:
                final_batch_scores += (0.1 * self.pop_scores.reshape(1, -1))
            top_k = np.argpartition(final_batch_scores, -k, axis=1)[:, -k:]
            batch_preds = [idx[np.argsort(final_batch_scores[i][idx])[::-1]] for i, idx in enumerate(top_k)]
            predictions.extend(batch_preds)
        return np.array(predictions)


# ==============================================================================
# EXPÉRIENCE 10 : Sequential / Co-visitation
# Hypothèse : L'ordre de lecture compte (Tome 1 -> Tome 2).
# Résultat : SUCCÈS. Capture des patterns temporels forts manqués par le collaboratif classique.
# ==============================================================================
class SequentialRecommender(BaseRecommender):
    def __init__(self, n_users, n_items):
        super().__init__(n_users, n_items)
        self.transition_matrix = None
        self.short_term_user_matrix = None

    def fit(self, df_interactions, df_items, half_life_days=1):
        print("Fitting EXP: Sequential/Co-visitation Model...")
        
        # 1. Profil Court Terme (Derniers items vus)
        df = df_interactions.copy()
        df['last_user_ts'] = df.groupby('u_idx')['t'].transform('max')
        # Decay très rapide (1 jour) pour ne garder que l'actuel
        df['weight'] = np.exp(-(np.log(2) / half_life_days) * ((df['last_user_ts'] - df['t']) / (24 * 3600)))
        
        self.short_term_user_matrix = sparse.csr_matrix(
            (df['weight'], (df['u_idx'], df['i_idx'])), 
            shape=(self.n_users, self.n_items)
        )
        
        # 2. Matrice de Transition (Item -> Next Item)
        df_sorted = df.sort_values(['u_idx', 't'])
        df_sorted['next_i'] = df_sorted.groupby('u_idx')['i_idx'].shift(-1)
        df_seq = df_sorted.dropna(subset=['next_i'])
        
        transitions = df_seq.groupby(['i_idx', 'next_i']).size().reset_index(name='count')
        transitions['score'] = np.log1p(transitions['count'])
        
        row = transitions['i_idx'].values
        col = transitions['next_i'].values.astype(int)
        data = transitions['score'].values
        
        self.transition_matrix = sparse.csr_matrix(
            (data, (row, col)), 
            shape=(self.n_items, self.n_items)
        )

    def predict(self, k=10, batch_size=1000):
        predictions = []
        for start in range(0, self.n_users, batch_size):
            end = min(start + batch_size, self.n_users)
            
            last_items = self.short_term_user_matrix[start:end]
            scores = last_items.dot(self.transition_matrix)
            
            if sparse.issparse(scores): scores = scores.toarray()
            
            top_k = np.argpartition(scores, -k, axis=1)[:, -k:]
            batch_preds = [idx[np.argsort(scores[i][idx])[::-1]] for i, idx in enumerate(top_k)]
            predictions.extend(batch_preds)
            
        return np.array(predictions)
