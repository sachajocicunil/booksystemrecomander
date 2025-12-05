import os
import sys
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

# Imports
sys.path.append('.')
from src.models import SemanticHybridRecommender
from src.models.svd import SVDRecommender
from src.preprocessing import DataLoader
from src.metrics import mapk_score

# CONFIG
THIS_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', 'data'))

print("🚀 Lancement Ensemble Tuning (Hybrid + SVD)...")
loader = DataLoader(f'{DATA_DIR}/interactions_train.csv', f'{DATA_DIR}/items.csv')
train_df, val_df = loader.get_time_split(train_ratio=0.8)

# Validation Matrix
val_rows = val_df['u_idx'].values
val_cols = val_df['i_idx'].values
val_data = np.ones(len(val_df))
val_matrix = sparse.csr_matrix(
    (val_data, (val_rows, val_cols)),
    shape=(loader.n_users, loader.n_items)
)

# 1. Train Hybrid (Optimized)
print("Training Hybrid Model...")
hybrid = SemanticHybridRecommender(loader.n_users, loader.n_items)
hybrid.fit(train_df, loader.items_df, alpha=0.5, half_life_days=[1, 250], ensemble_weights=[0.5, 0.5])

# 2. Train SVD
print("Training SVD Model...")
svd = SVDRecommender(loader.n_users, loader.n_items, n_factors=100)
svd.fit(train_df)

# Normalize SVD vectors for consistent scale
svd.user_vecs = normalize(svd.user_vecs, axis=1)
svd.item_vecs = normalize(svd.item_vecs.T, axis=1).T # Transpose to normalize items (features)

# 3. Ensemble Prediction & Scoring
def evaluate_ensemble(weight_hybrid):
    predictions = []
    batch_size = 1000
    
    for start_idx in range(0, loader.n_users, batch_size):
        end_idx = min(start_idx + batch_size, loader.n_users)
        
        # Get Scores
        scores_h = hybrid.get_batch_scores(start_idx, end_idx, re_buy_factor=0.5, pop_factor=0.2)
        scores_s = svd.get_scores_batch(start_idx, end_idx)
        
        # Normalize batch-wise to be safe? 
        # Hybrid scores are ~0-2. SVD scores (cosine) are ~ -1 to 1.
        # Shift SVD to 0-1 range? (x + 1)/2
        scores_s = (scores_s + 1) / 2
        
        # Ensemble
        scores_final = (scores_h * weight_hybrid) + (scores_s * (1 - weight_hybrid))
        
        # Top K
        k = 10
        top_k_unsorted = np.argpartition(scores_final, -k, axis=1)[:, -k:]
        
        batch_preds = []
        for i in range(len(scores_final)):
            row_scores = scores_final[i]
            idx = top_k_unsorted[i]
            sorted_idx = idx[np.argsort(row_scores[idx])[::-1]]
            batch_preds.append(sorted_idx)
        predictions.extend(batch_preds)
        
    score = mapk_score(np.array(predictions), val_matrix, k=10)
    return score

# Grid Search Weights
weights = [1.0, 0.95, 0.9, 0.8, 0.7]
print(f"\n--- Testing Ensemble Weights ---")

best_score = -1
best_w = -1

for w in weights:
    score = evaluate_ensemble(w)
    print(f"Weight Hybrid={w}, SVD={1-w:.2f} -> MAP@10: {score:.5f}")
    if score > best_score:
        best_score = score
        best_w = w

print(f"\n🏆 Best Ensemble: Hybrid={best_w}, SVD={1-best_w:.2f}, MAP={best_score:.5f}")
