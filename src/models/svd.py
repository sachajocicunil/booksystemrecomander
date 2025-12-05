import numpy as np
from scipy.sparse import linalg
from scipy import sparse
from .base import BaseRecommender

class SVDRecommender(BaseRecommender):
    """
    Recommandeur basé sur la SVD (Singular Value Decomposition) de la matrice d'interactions.
    Capture les facteurs latents globaux (Patterns cachés).
    """
    def __init__(self, n_users, n_items, n_factors=50):
        super().__init__(n_users, n_items)
        self.n_factors = n_factors
        self.user_vecs = None
        self.item_vecs = None
        
    def fit(self, df_interactions, df_items=None, **kwargs):
        """
        Factorise la matrice d'interactions R ~ U * V.T
        """
        print(f"Fitting SVD (Factors={self.n_factors})...")
        
        # 1. Construction Matrice Sparse
        # On binarise ou on utilise des poids ? Ici binaire simple ou count
        # Pour SVD, log(1+count) est souvent bien.
        
        # Déduplication pour la matrice
        df_grp = df_interactions.groupby(['u_idx', 'i_idx']).size().reset_index(name='count')
        
        row = df_grp['u_idx'].values
        col = df_grp['i_idx'].values
        data = np.log1p(df_grp['count'].values) # Log-count weighting
        
        # Matrice sparse (float pour SVD)
        R_sparse = sparse.csr_matrix((data, (row, col)), shape=(self.n_users, self.n_items), dtype=float)
        
        # 2. SVDs (Sparse SVD)
        # k = n_factors
        # u: (n_users, k), s: (k,), vt: (k, n_items)
        u, s, vt = linalg.svds(R_sparse, k=self.n_factors)
        
        # On intègre s dans u ou vt pour simplifier le dot product
        # U_final = u * sqrt(s)
        # V_final = vt.T * sqrt(s)
        # Mais svds renvoie s trié croissant, on doit inverser pour avoir les top components ?
        # svds renvoie les plus petites valeurs singulières par défaut ? NON, 'LM' (Largest Magnitude) par défaut pour k.
        # Mais l'ordre de retour est du plus petit au plus grand k.
        
        # On inverse pour avoir les plus importants en premier (convention, pas strict requis pour le dot)
        u = u[:, ::-1]
        s = s[::-1]
        vt = vt[::-1, :]
        
        s_diag = np.diag(np.sqrt(s))
        
        self.user_vecs = u @ s_diag
        self.item_vecs = s_diag @ vt
        
        print("SVD Fitted.")

    def predict(self, k=10, batch_size=1000):
        """
        Score = User_Vec . Item_Vec
        """
        predictions = []
        
        # Item vecs: (k, n_items)
        # User vecs: (n_users, k)
        
        for start_idx in range(0, self.n_users, batch_size):
            end_idx = min(start_idx + batch_size, self.n_users)
            
            u_batch = self.user_vecs[start_idx:end_idx] # (batch, k)
            
            # Score (batch, n_items)
            scores = np.dot(u_batch, self.item_vecs)
            
            # Top K
            top_k_unsorted = np.argpartition(scores, -k, axis=1)[:, -k:]
            
            batch_preds = []
            for i in range(len(scores)):
                row_scores = scores[i]
                idx = top_k_unsorted[i]
                sorted_idx = idx[np.argsort(row_scores[idx])[::-1]]
                batch_preds.append(sorted_idx)
                
            predictions.extend(batch_preds)
            
        return np.array(predictions)
    
    def get_scores_batch(self, u_idx_start, u_idx_end):
        """Retourne les scores bruts pour un batch (utile pour l'ensemble)"""
        u_batch = self.user_vecs[u_idx_start:u_idx_end]
        return np.dot(u_batch, self.item_vecs)
